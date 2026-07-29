import torch
import torch.nn as nn
import config


# ==========================================================
# CNN Feature Extractor
# ==========================================================

class CNNBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 32 x 128
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 16 x 64
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 8 x 32
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((2,1), (2,1)),

            # 4 x 32
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((2,1),(2,1)),

            # 2 x 32
            nn.Conv2d(512,512,2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)

        )

    def forward(self,x):

        return self.features(x)
    # ==========================================================
# Bidirectional LSTM
# ==========================================================

class BidirectionalLSTM(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True
        )

        self.embedding = nn.Linear(
            hidden_size*2,
            output_size
        )

    def forward(self,x):

        recurrent,_ = self.rnn(x)

        T,B,H = recurrent.size()

        recurrent = recurrent.reshape(T*B,H)

        output = self.embedding(recurrent)

        output = output.reshape(T,B,-1)

        return output
    # ==========================================================
# CRNN
# ==========================================================

class CRNN(nn.Module):

    def __init__(self,num_classes):

        super().__init__()

        self.cnn = CNNBackbone()

        self.rnn = nn.Sequential(

            BidirectionalLSTM(
                512,
                config.HIDDEN_SIZE,
                config.HIDDEN_SIZE
            ),

            BidirectionalLSTM(
                config.HIDDEN_SIZE,
                config.HIDDEN_SIZE,
                num_classes
            )

        )

    def forward(self,x):

        conv = self.cnn(x)

        b,c,h,w = conv.size()

        assert h == 1, f"Expected height = 1, got {h}"

        conv = conv.squeeze(2)

        conv = conv.permute(2,0,1)

        output = self.rnn(conv)

        return output
    # ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    NUM_CLASSES = 78

    model = CRNN(NUM_CLASSES)

    x = torch.randn(4,1,32,128)

    y = model(x)

    print(model)

    print()

    print("Input :",x.shape)

    print("Output :",y.shape)