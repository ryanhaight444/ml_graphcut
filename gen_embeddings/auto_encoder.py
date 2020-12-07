import argparse
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

def save_frame(save_dir,frame,i):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir+'frame' + str(i).zfill(6), frame)

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 5
batch_size = 50
learning_rate = 1e-3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--frame_dir', required=True, type=str,
                        help='a path to a video to generate pixel embeddings on')
    parser.add_argument('--new_frame_dir', required=True, type=str,
                        help='The directory to save the projected embeddings')
    parser.add_argument('--new_size', required=True, type=int,
                        help='The size that we want the embeddings to be' +
                        ' projected to')

    args = parser.parse_args()






img_transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class embeddings_dataset:
    frame_dir = None
    num_frames = None
    dims = None
        
    cur_frame = None
    frame_index = 0
    index = 0    
    
    index_arr = []
    max_index = 0

    frame_index_arr = []
    
    def __init__(self, frame_dir):
        self.frame_dir = frame_dir

        self.num_frames = len(os.listdir(self.frame_dir))
                 
        self.frame_index_arr = np.arange(self.num_frames)
        np.random.shuffle(self.frame_index_arr)
        
        self.cur_frame = np.load(self.frame_dir+'frame' + str(self.frame_index_arr[0]).zfill(6
)+"_embeddings.npy")  

        self.dims = self.cur_frame.shape
        
        self.index_arr = np.arange(self.dims[0] * self.dims[1])
        np.random.shuffle(self.index_arr)
        
        self.max_index = (self.dims[0] * self.dims[1])
                
    

    def __len__(self):
        return(self.num_frames * self.dims[0] * self.dims[1])

    def __getitem__(self, key):
        #frame_index,r = divmod(key, self.dims[0]* self.dims[1])
        #row,col = divmod(r, self.dims[1])
        #frame = np.load(self.frame_dir+'frame' + str(frame_index).zfill(6)+"_embeddings.npy")        
        #pixel_embedding = frame[row][col]
        if(self.index>=self.max_index):
            self.index = 0
            np.random.shuffle(self.index_arr)
            
            self.frame_index += 1
                   
            if(self.frame_index >= self.num_frames):
                np.random.shuffle(self.frame_index_arr)
                self.frame_index = 0

            self.cur_frame = np.load(self.frame_dir+'frame' + str(self.frame_index_arr[self.frame_index]).zfill(6)+"_embeddings.npy")





        row,col = divmod(self.index_arr[self.index], self.dims[1])
        pixel_embedding = self.cur_frame[row][col]    

        self.index += 1

        return(pixel_embedding)

dataset = embeddings_dataset(args.frame_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


class autoencoder(nn.Module):
    def __init__(self,args):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4096, args.new_size),
            nn.ReLU(True)

        )
        self.decoder = nn.Sequential(
            nn.Linear(args.new_size, 4096),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def gen_embeddings(self,args,x):
        y = np.zeros((x.shape[0], x.shape[1], args.new_size))
        for i,row in enumerate(x):
            for j,pixel in enumerate(row):
                pixel = torch.from_numpy(pixel).cuda()
                #pixel = Variable(pixel).cuda()
                y[i][j] = self.encoder(pixel).detach().cpu().numpy()
        print(y.shape)
        return y


model = autoencoder(args).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)



for epoch in range(num_epochs):
    for data in dataloader:
        img = data

        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.item()))
    #if epoch % 10 == 0:
        #pic = to_img(output.cpu().data)
        #save_image(pic, './dc_img/image_{}.png'.format(epoch))
    
for i,frame in enumerate(os.listdir(args.frame_dir)):

    frame = np.load(args.frame_dir+ frame)
    frame = model.gen_embeddings(args,frame)
    save_frame(args.new_frame_dir, frame, i)
torch.save(model.state_dict(), './conv_autoencoder.pth')
