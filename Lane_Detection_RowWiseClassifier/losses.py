import torch
import torch.nn as nn
import torch.nn.functional as F

class E2ELoss(nn.Module):
    def __init__(self, lambda_vertx = 1.0, lambda_lane = 1.0):
        super().__init__()
        self.lambda_vertx = lambda_vertx
        self.lambda_lane= lambda_lane

        # Binary loss
        self.bce = nn.BCEWithLogitsLoss()
        # Location classification
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, lane_exist_out, vertex_outputs, location_outputs, lane_exist, vertex_exist, lane_x ):

        # Lane marker existance loss
        lane_loss = self.bce(lane_exist_out, lane_exist)

        # vertex wise existance loss
        # vertex_outputs = [vertex_wise_confidence_out_1, vertex_wise_confidence_out_2,..]
        vertex_loss = 0.0
        for lane_id in range(len(vertex_outputs)):
            prediction = vertex_outputs[lane_id]
            target = vertex_exist[:,lane_id,:]
            vertex_loss += self.bce(prediction, target)
        vertex_loss /= len(vertex_outputs) # average over lanes

        # row wise location loss
        # location_outputs = [row_wise_vertex_location_out_1,row_wise_vertex_location_out_2,...]
        location_loss = 0.0
        valid_lane_count = 0
        for lane_id in range(len(location_outputs)):
            prediction = location_outputs[lane_id] # [B, 256, 128]
            target= lane_x[:,lane_id,:] # [B, 128]
            ce_loss = self.ce(prediction, target) # [B, 128]
            # Only use rows where a vertex actually exists
            valid = vertex_exist[:,lane_id,:]
            # keep only valid rows
            valid_loss = (ce_loss* valid)
            # Normalize by number of valid rows
            num_valid = valid.sum()
            if num_valid>0:
                lane_location_loss = (valid_loss.sum()/num_valid)
                location_loss += (lane_location_loss)
                valid_lane_count+=1
        if valid_lane_count>0: # Average across lanes that contain vertices
            location_loss /= valid_lane_count
        else:
            location_loss = location_outputs[0].sum()* 0.0 #

        # Total loss
        total_loss = (location_loss + self.lambda_vertx*vertex_loss + self.lambda_lane*lane_loss)
        
        return {
            "total": total_loss,
            "location": location_loss,
            "vertex": vertex_loss,
            "lane": lane_loss
        }

