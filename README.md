# Explainable AI in MuZero
This branch contains a pre-trained model (model.checkpoint) of breakout and cartpole for explanation.

## Usage
### Install required packages
```
pip install -r requirements.txt
```
### Checkpoint location
Checkpoints are stored in "results/[game name]"
### Run
```bash
python muzero.py
```
Enter 1 to choose Breakout, enter  load the pre-trained model for Breakout (ID=1).
Then choose:
- 3 to render self-play games
- 4 to play against MuZero agents

### Run debugger
- Only work with Ubuntu OS
- Install ray nightly-dev version (Download wheel at: https://docs.ray.io/en/master/installation.html)
```
pip install -U [link to wheel]
```  
- Stick below code as a checkpoint
```
ray.util.pdb.set_trace()
```
- When run the program, open another terminal and run following command:
```angular2html
ray debug
```

### Check the Tensorboard
The online tensorboard will be updated from server at: https://tensorboard.dev/experiment/TBcg1CejT2WZb4V8fyA6AA/ for further information of the model and training system.

### Model architecture
- Resnet of Breakout:
```
MuZeroResidualNetwork(

(representation_network): DataParallel(

(module): RepresentationNetwork(

  (downsample_net): DownSample(

    (conv1): Conv2d(3, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    (resblocks1): ModuleList(

      (0): ResidualBlock(

        (conv1): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        (conv2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      )

      (1): ResidualBlock(

        (conv1): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        (conv2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      )

    )

    (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    (resblocks2): ModuleList(

      (0): ResidualBlock(

        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      )

      (1): ResidualBlock(

        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      )

      (2): ResidualBlock(

        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      )

    )

    (pooling1): AvgPool2d(kernel_size=3, stride=2, padding=1)

    (resblocks3): ModuleList(

      (0): ResidualBlock(

        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      )

      (1): ResidualBlock(

        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      )

      (2): ResidualBlock(

        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      )

    )

    (pooling2): AvgPool2d(kernel_size=3, stride=2, padding=1)

  )

  (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

  (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

  (resblocks): ModuleList(

    (0): ResidualBlock(

      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    )

    (1): ResidualBlock(

      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    )

  )

)
)

(dynamics_network): DataParallel(

(module): DynamicsNetwork(

  (conv): Conv2d(17, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

  (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

  (resblocks): ModuleList(

    (0): ResidualBlock(

      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    )

    (1): ResidualBlock(

      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    )

  )

  (conv1x1_reward): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1))

  (fc): Sequential(

    (0): Linear(in_features=144, out_features=16, bias=True)

    (1): ELU(alpha=1.0)

    (2): Linear(in_features=16, out_features=21, bias=True)

    (3): Identity()

  )

)
)

(prediction_network): DataParallel(

(module): PredictionNetwork(

  (resblocks): ModuleList(

    (0): ResidualBlock(

      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    )

    (1): ResidualBlock(

      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    )

  )

  (conv1x1_value): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1))

  (conv1x1_policy): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1))

  (fc_value): Sequential(

    (0): Linear(in_features=144, out_features=16, bias=True)

    (1): ELU(alpha=1.0)

    (2): Linear(in_features=16, out_features=21, bias=True)

    (3): Identity()

  )

  (fc_policy): Sequential(

    (0): Linear(in_features=144, out_features=16, bias=True)

    (1): ELU(alpha=1.0)

    (2): Linear(in_features=16, out_features=4, bias=True)

    (3): Identity()

  )

)
)

)
```
