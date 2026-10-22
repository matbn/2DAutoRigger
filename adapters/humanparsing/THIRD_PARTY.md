# Third party material

This package is **adapter code**, not model weights. No weights are committed
to this repository and none are downloaded for you.

| What | Where | Terms |
| --- | --- | --- |
| Human parsing (LIP / ATR labels) | [https://github.com/GoGoBunny/HRNet-Semantic-Segmentation](https://github.com/GoGoBunny/HRNet-Semantic-Segmentation) | MIT -- see https://github.com/GoGoBunny/HRNet-Semantic-Segmentation |
| The adapter code in this repository | this repository | Apache-2.0 |

Parsing models are trained on human photographs and datasets such as LIP and ATR. Check the dataset licenses before commercial use.

## What you have to do

1. Install the requirements: `pip install -r requirements.txt`.
2. Obtain the weights yourself, from the source above.
3. Read the terms linked above and make sure they are acceptable to you.
4. Point the adapter at the weights, either with the environment variable
   `RIGGER2D_HUMANPARSING_WEIGHTS` or by constructing the adapter with an explicit path.

The core of 2DAutoRigger never sees, downloads or redistributes these weights.
It only asks the adapter whether it is ready.
