/**
 * The mask and joint editor.
 *
 * The canvas keeps its own label map -- one byte per pixel saying which part
 * it belongs to -- so that painting is instant. The server keeps the edit log
 * and is the source of truth; every stroke is sent to it, and the masks are
 * read back afterwards.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { Document, Edit, Vec2 } from "@rigger2d/schema";

export type BrushMode = "add" | "erase" | "assign";

export interface PartColour {
  id: string;
  colour: string;
}

export interface EditorProps {
  document: Document;
  landmarks: Record<string, Vec2>;
  client: {
    maskUrl: (id: string, partId: string) => string;
  };
  parts: PartColour[];
  activePart: string;
  brush: BrushMode;
  brushRadius: number;
  onMaskEdit: (edit: Edit) => void;
  onJointEdit: (edit: Edit) => void;
  reloadToken: number;
}

const LANDMARK_ORDER = [
  "head_top",
  "head_center",
  "neck",
  "shoulder_l",
  "shoulder_r",
  "elbow_l",
  "elbow_r",
  "wrist_l",
  "wrist_r",
  "hip_l",
  "hip_r",
  "knee_l",
  "knee_r",
  "ankle_l",
  "ankle_r",
];

export function Editor(props: EditorProps) {
  const { document, parts, activePart, brush, brushRadius } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const labelsRef = useRef<Uint8Array | null>(null);
  const strokesRef = useRef<Vec2[]>([]);
  const dragRef = useRef<{ name: string; origin: Vec2; start: Vec2 } | null>(null);
  const [localLandmarks, setLocalLandmarks] = useState<Record<string, Vec2>>({});
  const [loaded, setLoaded] = useState(false);

  const width = document.image.width;
  const height = document.image.height;
  const partIndex = new Map(parts.map((part, index) => [part.id, index + 1]));

  // The landmarks as they should be drawn: the server's, plus whatever the
  // user is dragging right now.
  const landmarks = { ...props.landmarks, ...localLandmarks };

  // -- load the masks into a label map ---------------------------------
  useEffect(() => {
    let cancelled = false;
    const partIds = document.parts?.masks.map((mask) => mask.part_id) ?? [];
    if (partIds.length === 0) {
      labelsRef.current = null;
      setLoaded(false);
      return () => {
        cancelled = true;
      };
    }

    const labels = new Uint8Array(width * height);
    const canvas = globalThis.document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      return () => {
        cancelled = true;
      };
    }

    const loadOne = async (partId: string, index: number): Promise<void> => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.src = props.client.maskUrl(document.project_id, partId);
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error(`could not load the mask for ${partId}`));
      });
      if (cancelled) return;
      context.clearRect(0, 0, width, height);
      context.drawImage(image, 0, 0, width, height);
      const data = context.getImageData(0, 0, width, height).data;
      for (let pixel = 0; pixel < width * height; pixel += 1) {
        if ((data[pixel * 4 + 3] ?? 0) > 127) {
          labels[pixel] = index;
        }
      }
    };

    void (async () => {
      try {
        // Later parts are drawn on top, so they are applied last.
        for (const [index, partId] of partIds.entries()) {
          await loadOne(partId, index + 1);
        }
        if (cancelled) return;
        labelsRef.current = labels;
        setLoaded(true);
      } catch {
        if (!cancelled) setLoaded(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [document.parts, document.project_id, height, width, props.client, props.reloadToken]);

  // -- draw -------------------------------------------------------------
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    context.clearRect(0, 0, width, height);
    const image = imageRef.current;
    if (image) {
      context.drawImage(image, 0, 0, width, height);
    }

    const labels = labelsRef.current;
    if (labels) {
      const overlay = context.createImageData(width, height);
      for (let pixel = 0; pixel < width * height; pixel += 1) {
        const index = labels[pixel] ?? 0;
        if (index === 0) continue;
        const part = parts[index - 1];
        if (!part) continue;
        const rgb = hexToRgb(part.colour);
        overlay.data[pixel * 4 + 0] = rgb[0];
        overlay.data[pixel * 4 + 1] = rgb[1];
        overlay.data[pixel * 4 + 2] = rgb[2];
        overlay.data[pixel * 4 + 3] = 110;
      }
      context.putImageData(overlay, 0, 0);
    }

    // The joints.
    context.save();
    context.fillStyle = "#ffffff";
    context.strokeStyle = "#101018";
    context.lineWidth = 2;
    for (const name of LANDMARK_ORDER) {
      const point = landmarks[name];
      if (!point) continue;
      context.beginPath();
      context.arc(point.x, point.y, 6, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.fillStyle = "#101018";
      context.font = "11px system-ui, sans-serif";
      context.fillText(name, point.x + 8, point.y - 6);
      context.fillStyle = "#ffffff";
    }
    context.restore();
  }, [height, landmarks, parts, width]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = `/api/projects/${document.project_id}/source`;
    image.onload = () => {
      imageRef.current = image;
      draw();
    };
    draw();
  }, [document.project_id, draw]);

  useEffect(() => {
    draw();
  }, [draw, loaded]);

  // -- painting ---------------------------------------------------------
  const paint = useCallback(
    (point: Vec2) => {
      const labels = labelsRef.current;
      if (!labels) return;
      const index = partIndex.get(activePart) ?? 0;
      const radius = brushRadius;
      const x0 = Math.max(0, Math.floor(point.x - radius));
      const x1 = Math.min(width - 1, Math.ceil(point.x + radius));
      const y0 = Math.max(0, Math.floor(point.y - radius));
      const y1 = Math.min(height - 1, Math.ceil(point.y + radius));
      for (let y = y0; y <= y1; y += 1) {
        for (let x = x0; x <= x1; x += 1) {
          const dx = x - point.x;
          const dy = y - point.y;
          if (dx * dx + dy * dy > radius * radius) continue;
          const pixel = y * width + x;
          if (brush === "erase") {
            labels[pixel] = 0;
          } else if (brush === "assign") {
            labels[pixel] = index;
          } else {
            labels[pixel] = index || labels[pixel]!;
          }
        }
      }
      draw();
    },
    [activePart, brush, brushRadius, draw, height, partIndex, width],
  );

  const canvasPoint = (event: React.PointerEvent<HTMLCanvasElement>): Vec2 => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * width,
      y: ((event.clientY - rect.top) / rect.height) * height,
    };
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>): void => {
    const point = canvasPoint(event);
    const near = LANDMARK_ORDER.find((name) => {
      const landmark = landmarks[name];
      if (!landmark) return false;
      return Math.hypot(landmark.x - point.x, landmark.y - point.y) < 10;
    });
    if (near) {
      const origin = landmarks[near];
      if (origin) {
        dragRef.current = { name: near, origin, start: point };
      }
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    strokesRef.current = [point];
    paint(point);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>): void => {
    const point = canvasPoint(event);
    const drag = dragRef.current;
    if (drag) {
      setLocalLandmarks((current) => ({
        ...current,
        [drag.name]: { x: point.x, y: point.y },
      }));
      return;
    }
    if (strokesRef.current.length === 0) return;
    strokesRef.current.push(point);
    paint(point);
  };

  const onPointerUp = (): void => {
    const drag = dragRef.current;
    if (drag) {
      const moved = localLandmarks[drag.name];
      dragRef.current = null;
      setLocalLandmarks({});
      if (moved) {
        props.onJointEdit({
          joint: {
            joint_id: drag.name,
            dx: moved.x - drag.origin.x,
            dy: moved.y - drag.origin.y,
          },
        });
      }
      return;
    }
    const points = strokesRef.current;
    strokesRef.current = [];
    if (points.length === 0) return;
    props.onMaskEdit({
      mask: { op: brush, part_id: activePart, points, radius: brushRadius },
    });
  };

  return (
    <canvas
      ref={canvasRef}
      className="editor-canvas"
      width={width}
      height={height}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={onPointerUp}
    />
  );
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  const value = Number.parseInt(clean, 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}
