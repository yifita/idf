from network.network import Network
from pykdtree.kdtree import KDTree
from network.siren import gradient
from helper import AllOf
import torch
import sys
class Renderer(Network):

    def __init__(self, config):
        self.base : Network = None
        self.batch_size = 125000
        self.epsilon = 1e-4
        self.up = AllOf([1,0,0])
        self.direction = AllOf([0,1,0])
        self.tracing_steps = 100
        #middle of the camera plane
        self.camera = AllOf([0,-0.99,0])
        self.res = AllOf([1024,1024])
        self.freeze_base = False
        self.alpha = 1.5
        self.alpha_inner = 1

        super().__init__(config)

        self.camera = torch.FloatTensor(self.camera.elements)
        self.up = torch.FloatTensor(self.up.elements)
        self.direction = torch.FloatTensor(self.direction.elements)

        #asuming direction is normalized this should be the same factor as up
        self.left = torch.cross(self.up,self.direction)
        self.res = self.res.elements
        self.total = self.res[0] * self.res[1]
        xSteps = torch.linspace(-1,1,self.res[0])
        ySteps = torch.linspace(-1,1,self.res[1])
        xSteps,ySteps = torch.meshgrid(xSteps,ySteps)

        offsets = xSteps.unsqueeze(-1)*self.left.reshape([1,1,3]) + ySteps.unsqueeze(-1) * self.up.reshape([1,1,3])
        offsets += self.camera.reshape([1,1,3])
        self.image_points = offsets.reshape([-1,3]).cuda()

        self.camera = self.camera.cuda()
        self.up = self.up.cuda()
        self.direction = self.direction.cuda()
        self.pdist = torch.nn.PairwiseDistance().cuda()
        self.alpha = torch.FloatTensor([self.alpha]).cuda()
        self.alpha_inner = torch.FloatTensor([self.alpha_inner]).cuda()
        if(self.freeze_base):
             for param in self.base.parameters():
                param.requires_grad = False


    def fake_sdf(self,coords):
        return (coords.norm(dim=-1) - 0.5).reshape([-1,1])

    def forward(self, args):

        num_samples = self.image_points.shape[0]
        num_batches = (num_samples + self.batch_size-1)//self.batch_size

        use_mask = torch.ones([num_samples],dtype=torch.int32)
        count_mask = torch.zeros([num_samples],dtype=torch.int32)
        final_sdf =  torch.ones([num_samples]).cuda()
        direction = self.direction.reshape([-1,3])
        final_points = self.image_points.clone()

        for i in range(self.tracing_steps):
            #filter out of bounds
            torch.cuda.empty_cache()
            index = torch.where(use_mask)[0]
            index_len = len(index)
            print(index_len)
            if index_len == 0:
                break
            if(index_len > self.batch_size):
                index = index[:self.batch_size]
            count_mask[index] += 1
            pts = final_points[index,:]
            torch.cuda.empty_cache()
            sdf = self.base({"coords":final_points[index],"detach":False})["sdf"].detach()
            pts += direction*sdf*torch.where(sdf < 0,self.alpha_inner,self.alpha)
            final_sdf[index] = sdf.flatten()
            final_points[index,:] = pts
            use_mask[count_mask > 1000] = 0
            use_mask[torch.abs(final_sdf)<self.epsilon] = 0
            use_mask[torch.max(torch.abs(final_points),dim=-1)[0] > 1] = 0


        result = torch.where(torch.abs(final_sdf)>self.epsilon,torch.zeros_like(final_sdf),self.pdist(final_points,self.image_points))

        final_sdf = final_sdf.reshape(self.res[0],self.res[1])
        result = result.reshape(self.res[0],self.res[1])
        self.runner.logger.log_image("final_sdf",final_sdf.detach())
        self.runner.logger.log_image("image",result.detach())
        self.runner.logger.log_hist("himage",result.detach().cpu())
        self.runner.logger.log_hist("hsdf",final_sdf.detach().cpu())
        self.runner.logger.log_hist("points",self.image_points.detach().cpu())
        self.runner.logger.log_mesh("points",final_points.detach().cpu(),None)

        return {"final_sdf":final_sdf,"render":result,"final_points":final_points}

    def save(self, path):
        torch.save(self, path)


