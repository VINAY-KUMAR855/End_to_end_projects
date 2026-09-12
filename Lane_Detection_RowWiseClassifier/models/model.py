import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.ERFNet import ERFNet
from models.HRM import HRM_Block

class E2ENet(nn.Module):
    def __init__(self,channels=96, nums_lane = 6, culomn_channels = 256, row_channels = 128, initialed = True):
        super().__init__()
        self.backbone = ERFNet(channels = 96)
        self.share_hrm1 = HRM_Block(in_planes=channels, stride=2, kernel_size=3) 
        self.share_hrm2 = HRM_Block(in_planes=channels, stride=2, kernel_size=3) 
        self.share_hrm3 = HRM_Block(in_planes=channels, stride=2, kernel_size=3) 

        self.lane_marker_confidence = nn.Sequential(
            nn.Linear(channels, channels//2),
            nn.BatchNorm1d(channels//2),
            nn.ReLU(inplace=True),
            nn.Linear(channels//2, nums_lane)
        )

        self.lane_marker_block1 = nn.Sequential(
            # maker_wise_hrms
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 16)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 8)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 4)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 2)
            HRM_Block(in_planes=channels, stride=2, kernel_size=1), # (B, 96, 128, 1)
            # vertex_wise_confidence_branch: gives whether the vertex exist or not in the lane
            nn.Conv2d(channels, channels//2, kernel_size=1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, kernel_size=1), # (B, 96, 128, 1) -> (B, 1, 128,1)
            # row_wise_vertex_location_branch
            nn.Conv2d(channels, row_channels, kernel_size=1), # (B, 96, 128, 1) -> (B, 128, 128, 1)
            nn.BatchNorm2d(row_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(row_channels, culomn_channels, kernel_size=1) # (B, 128, 128, 1) -> (B, 256, 128, 1)
        )
        self.lane_marker_block2 = nn.Sequential(
            # maker_wise_hrms
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 16)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 8)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 4)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 2)
            HRM_Block(in_planes=channels, stride=2, kernel_size=1), # (B, 96, 128, 1)
            # vertex_wise_confidence_branch: gives whether the vertex exist or not in the lane
            nn.Conv2d(channels, channels//2, kernel_size=1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, kernel_size=1), # (B, 96, 128, 1) -> (B, 1, 128,1)
            # row_wise_vertex_location_branch
            nn.Conv2d(channels, row_channels, kernel_size=1), # (B, 96, 128, 1) -> (B, 128, 128, 1)
            nn.BatchNorm2d(row_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(row_channels, culomn_channels, kernel_size=1) # (B, 128, 128, 1) -> (B, 256, 128, 1)
        )
        self.lane_marker_block3 = nn.Sequential(
            # maker_wise_hrms
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 16)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 8)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 4)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 2)
            HRM_Block(in_planes=channels, stride=2, kernel_size=1), # (B, 96, 128, 1)
            # vertex_wise_confidence_branch: gives whether the vertex exist or not in the lane
            nn.Conv2d(channels, channels//2, kernel_size=1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, kernel_size=1), # (B, 96, 128, 1) -> (B, 1, 128,1)
            # row_wise_vertex_location_branch
            nn.Conv2d(channels, row_channels, kernel_size=1), # (B, 96, 128, 1) -> (B, 128, 128, 1)
            nn.BatchNorm2d(row_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(row_channels, culomn_channels, kernel_size=1) # (B, 128, 128, 1) -> (B, 256, 128, 1)
        )
        self.lane_marker_block4 = nn.Sequential(
            # maker_wise_hrms
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 16)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 8)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 4)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 2)
            HRM_Block(in_planes=channels, stride=2, kernel_size=1), # (B, 96, 128, 1)
            # vertex_wise_confidence_branch: gives whether the vertex exist or not in the lane
            nn.Conv2d(channels, channels//2, kernel_size=1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, kernel_size=1), # (B, 96, 128, 1) -> (B, 1, 128,1)
            # row_wise_vertex_location_branch
            nn.Conv2d(channels, row_channels, kernel_size=1), # (B, 96, 128, 1) -> (B, 128, 128, 1)
            nn.BatchNorm2d(row_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(row_channels, culomn_channels, kernel_size=1) # (B, 128, 128, 1) -> (B, 256, 128, 1)
        )
        self.lane_marker_block5 = nn.Sequential(
            # maker_wise_hrms
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 16)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 8)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 4)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 2)
            HRM_Block(in_planes=channels, stride=2, kernel_size=1), # (B, 96, 128, 1)
            # vertex_wise_confidence_branch: gives whether the vertex exist or not in the lane
            nn.Conv2d(channels, channels//2, kernel_size=1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, kernel_size=1), # (B, 96, 128, 1) -> (B, 1, 128,1)
            # row_wise_vertex_location_branch
            nn.Conv2d(channels, row_channels, kernel_size=1), # (B, 96, 128, 1) -> (B, 128, 128, 1)
            nn.BatchNorm2d(row_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(row_channels, culomn_channels, kernel_size=1) # (B, 128, 128, 1) -> (B, 256, 128, 1)
        )
        self.lane_marker_block6 = nn.Sequential(
            # maker_wise_hrms
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 16)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 8)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 4)
            HRM_Block(in_planes=channels, stride=2, kernel_size=3), # (B, 96, 128, 2)
            HRM_Block(in_planes=channels, stride=2, kernel_size=1), # (B, 96, 128, 1)
            # vertex_wise_confidence_branch: gives whether the vertex exist or not in the lane
            nn.Conv2d(channels, channels//2, kernel_size=1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, kernel_size=1), # (B, 96, 128, 1) -> (B, 1, 128,1)
            # row_wise_vertex_location_branch
            nn.Conv2d(channels, row_channels, kernel_size=1), # (B, 96, 128, 1) -> (B, 128, 128, 1)
            nn.BatchNorm2d(row_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(row_channels, culomn_channels, kernel_size=1) # (B, 128, 128, 1) -> (B, 256, 128, 1)
        )
        if initialed:
            self.initialize_weights()
            print("Initialize Neural Network Successfully")

        
    def forward(self, x):
        x = self.backbone(x) # (B, 96, 128, 256)
        # shared HRM's
        x = self.share_hrm1(x) # (B, 96, 128, 128)
        x = self.share_hrm2(x) # (B, 96, 128, 64)
        x = self.share_hrm3(x) # (B, 96, 128, 32)

        # Lane marker confidence branch
        # this checks whether the lane exist's or not.
        lane_marker_confidence_x = nn.AdaptiveAvgPool2d((1,1))(x) # (B, 96, 1, 1)
        lane_marker_confidence_x = lane_marker_confidence_x.view(x.size(0),-1) # (B, 96)
        lane_marker_confidence_x = self.lane_marker_confidence(lane_marker_confidence_x)

        # Lane marker branch
        # This finds the lane coordinates and confidence of lane present in that point
        branch_x = self.lane_marker_block1[:5](x) # (B, 96, 128, 1)
        vertex_wise_confidence_x_1 = self.lane_marker_block1[5:9](branch_x).flatten(1) # (B, 128)
        row_wise_vertex_location_x_1 = self.lane_marker_block1[9:](branch_x).squeeze(-1) # (B, 256, 128)

        branch_x = self.lane_marker_block2[:5](x)
        vertex_wise_confidence_x_2 = self.lane_marker_block2[5:9](branch_x).flatten(1)
        row_wise_vertex_location_x_2 = self.lane_marker_block2[9:](branch_x).squeeze(-1)
        
        branch_x = self.lane_marker_block3[:5](x)
        vertex_wise_confidence_x_3 = self.lane_marker_block3[5:9](branch_x).flatten(1)
        row_wise_vertex_location_x_3 = self.lane_marker_block3[9:](branch_x).squeeze(-1)
        
        branch_x = self.lane_marker_block4[:5](x)
        vertex_wise_confidence_x_4 = self.lane_marker_block4[5:9](branch_x).flatten(1)
        row_wise_vertex_location_x_4 = self.lane_marker_block4[9:](branch_x).squeeze(-1)

        branch_x = self.lane_marker_block5[:5](x)
        vertex_wise_confidence_x_5 = self.lane_marker_block5[5:9](branch_x).flatten(1)
        row_wise_vertex_location_x_5 = self.lane_marker_block5[9:](branch_x).squeeze(-1)

        branch_x = self.lane_marker_block6[:5](x)
        vertex_wise_confidence_x_6 = self.lane_marker_block6[5:9](branch_x).flatten(1)
        row_wise_vertex_location_x_6 = self.lane_marker_block6[9:](branch_x).squeeze(-1)


        return {
            "lane_exit_out": lane_marker_confidence_x,
            "vertex_wise_confidence_out_1": vertex_wise_confidence_x_1, 
            "row_wise_vertex_location_out_1": row_wise_vertex_location_x_1, 
            "vertex_wise_confidence_out_2": vertex_wise_confidence_x_2, 
            "row_wise_vertex_location_out_2": row_wise_vertex_location_x_2,
            "vertex_wise_confidence_out_3": vertex_wise_confidence_x_3, 
            "row_wise_vertex_location_out_3": row_wise_vertex_location_x_3,
            "vertex_wise_confidence_out_4": vertex_wise_confidence_x_4, 
            "row_wise_vertex_location_out_4": row_wise_vertex_location_x_4,
            "vertex_wise_confidence_out_5": vertex_wise_confidence_x_5, 
            "row_wise_vertex_location_out_5": row_wise_vertex_location_x_5,
            "vertex_wise_confidence_out_6": vertex_wise_confidence_x_6, 
            "row_wise_vertex_location_out_6": row_wise_vertex_location_x_6
        }
    
    def initialize_weights(self):
        for model in self.modules():
            self.real_init_weights(model)

    def real_init_weights(self, m):
        if isinstance(m, list):
            for mini_m in m:
                self.real_init_weights(mini_m)
        else:
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # Kaiming / He initialization
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0.0, std=0.01)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Module):
                for mini_m in m.children():
                    self.real_init_weights(mini_m)
            else:
                print('unkonwn module', m)

if __name__ == '__main__':
    x = torch.randn(size=(2, 3, 256, 512))
    model = E2ENet(channels=96)
    result = model(x)
    for name, value in result.items():
        print(name, value.shape)