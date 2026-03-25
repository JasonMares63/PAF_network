import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from PAF import *
from RealNVP import *
from torch.cuda.amp import autocast, GradScaler

class PAF_trainer(nn.Module):
    def __init__(self,
                 x_esm_dim,
                 hidden_dim=128,
                 output_dim=64,
                 n_layers = 4,
                 n_flows = 4,
                 percentile = 0.5,
                 lr = [5e-3] * 3,
                 l2_coef = [5e-4] * 3,
                 dropout = [0.2] * 3,
                 device ="cpu"):
        super().__init__()
        
        self.esm_dim = x_esm_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.lr = lr
        self.l2_coef = l2_coef
        self.GAT_dropout = dropout[0]
        self.model = DeepMultiScalePPI(esm_dim = self.esm_dim,
                                       hidden_dim = self.hidden_dim,
                                       num_scales = 2,
                                       proj_dim = self.output_dim,
                                       dropout = self.GAT_dropout,
                                       )
        self.optimizer =torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr[0],
            weight_decay=self.l2_coef[0]
        )
        self.best_state = None
        
        self.classifier_dropout = dropout[1]
        self.model2 = DeepScalePrediction(proj_dim = self.output_dim,
                                          dropout = self.classifier_dropout)
        self.optimizer2 =torch.optim.AdamW(
            self.model2.parameters(),
            lr=self.lr[1],
            weight_decay=self.l2_coef[1]
        )
        self.best_state2 = None
        self.cat_emb = self.model2.cat_emb
        
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.flow_dropout = dropout[2]
        self.percentile = percentile
        self.threshold = None
        self.model3 = RealNVP(self.output_dim*2, self.n_layers,
                              n_flows = self.n_flows,
                              hidden_dim = self.hidden_dim,dropout = self.flow_dropout)
        
        self.optimizer3 =torch.optim.AdamW(
        self.model3.parameters(),
            lr=self.lr[2],
            weight_decay=self.l2_coef[2]
        )
        self.best_state3 = None
        
        self.train_CL_history = []
        self.train_MSE_history = []
        self.train_NLL_history = []
        self.valid_NLL_history = []
        self.valid_CL_history = []
        self.valid_MSE_history = []
        self.valid_NLL_history = []
        self.best_CL = float("inf")
        self.best_MSE = float("inf")
        self.best_NLL = float("inf")
      
    def train_step_embeddings(self, prior_esm_embed, prior_edge_index,prior_edge_weights):
        
        self.model.train()
        self.optimizer.zero_grad()
        
        with autocast(self.device):
            loss = self.model(prior_esm_embed ,prior_edge_index,prior_edge_weights)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        #print("Global Prior Training Complete. Weights Frozen.")
        return loss
    
    def train_step_prediction(self, prior_esm_embed, prior_edge_index,prior_edge_weights):
        
        self.model2.train()
        self.optimizer2.zero_grad()
        
        u, v = prior_edge_index
        with autocast(self.device):
            with torch.no_grad():
                h = self.model.get_embeddings(prior_esm_embed, prior_edge_index ,prior_edge_weights)
            
            loss, _  = self.model2(h[u] ,h[v],prior_edge_weights)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1.0)
        self.optimizer2.step()
            
        return loss
     
    @torch.no_grad   
    def valid_step_embeddings(self,valid_esm_embed, valid_edge_index,valid_edge_weights):
        
        self.model.eval()
        loss = self.model(valid_esm_embed,
                          valid_edge_index,
                          valid_edge_weights)
        #print("Global Prior Training Complete. Weights Frozen.")
        return loss     
    
    @torch.no_grad    
    def valid_step_prediction(self,valid_esm_embed, valid_edge_index,valid_edge_weights):
        
        with torch.no_grad():
            h = self.model.get_embeddings(valid_esm_embed, valid_edge_index,valid_edge_weights)
        u, v = valid_edge_index
        self.model2.eval()
        loss, _ = self.model2(h[u] ,h[v],valid_edge_weights)
        return loss
            
    def train_embeddings(self,
                        esm_embed,
                        prior_edge_index ,prior_edge_weights,
                        valid_edge_index,valid_edge_weights,
                         epochs = 100, verbose=True):
        #best_loss = float("inf")
        if verbose:
            pbar = tqdm.tqdm(range(1,epochs+1),
                                desc="Training Progress", unit="epoch")

            for epoch in pbar:
                train_loss = self.train_step_embeddings(esm_embed, prior_edge_index ,prior_edge_weights)
                self.train_CL_history.append(train_loss.item())
                valid_loss = self.valid_step_embeddings(esm_embed, valid_edge_index,valid_edge_weights)
                self.valid_CL_history.append(valid_loss.item())
                if self.best_CL > valid_loss:
                    self.best_CL = valid_loss
                    self.best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                pbar.set_postfix({"Epoch": f"{epoch:4d}/{epochs}",
                                        "Tr": f"{train_loss:.4f}",
                                        "Vall": f"{valid_loss:.4f}"})
        else:
            for epoch in range(1,epochs+1):
                train_loss = self.train_step_embeddings(esm_embed, prior_edge_index ,prior_edge_weights)
                self.train_CL_history.append(train_loss.item())
                valid_loss = self.valid_step_embeddings(esm_embed, valid_edge_index,valid_edge_weights)
                self.valid_CL_history.append(valid_loss.item())
                if self.best_CL > valid_loss:
                    self.best_CL = valid_loss
                    self.best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
        
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            #print(f"\nRestored best model (val_nll={best_val:.4f})")
        return self.best_CL

    def train_predictor(self,
                         esm_embed,
                         prior_edge_index ,prior_edge_weights,
                         valid_edge_index,valid_edge_weights,
                         epochs = 100,verbose = True):
        assert self.best_state is not None, 'train emebeddings first'
        
        if verbose:
            pbar = tqdm.tqdm(range(1,epochs+1),
                                desc="Training Progress", unit="epoch")

            for epoch in pbar:
                train_loss = self.train_step_prediction(esm_embed, prior_edge_index ,prior_edge_weights)
                self.train_MSE_history.append(train_loss.item())
                valid_loss = self.valid_step_prediction(esm_embed, valid_edge_index,valid_edge_weights)
                self.valid_MSE_history.append(valid_loss.item())
                if self.best_MSE > valid_loss:
                    self.best_MSE = valid_loss
                    self.best_state2 = {
                        k: v.cpu().clone()
                        for k, v in self.model2.state_dict().items()
                    }
                pbar.set_postfix({"Epoch": f"{epoch:4d}/{epochs}",
                                        "Tr": f"{train_loss:.4f}",
                                        "Vall": f"{valid_loss:.4f}"})
        else:
            for epoch in range(1,epochs+1):
                train_loss = self.train_step_prediction(esm_embed, prior_edge_index ,prior_edge_weights)
                self.train_MSE_history.append(train_loss.item())
                valid_loss = self.valid_step_prediction(esm_embed, valid_edge_index,valid_edge_weights)
                self.valid_MSE_history.append(valid_loss.item())
                if self.best_MSE > valid_loss:
                    self.best_MSE = valid_loss
                    self.best_state2 = {
                        k: v.cpu().clone()
                        for k, v in self.model2.state_dict().items()
                    }
                    
        if self.best_state2 is not None:
            self.model2.load_state_dict(self.best_state2)
            #print(f"\nRestored best model (val_nll={best_val:.4f})")
        return self.best_MSE

    def _get_embeddings(self,x, edge_index=None,edge_weights= None):
        assert self.best_state is not None, 'train emebeddings first'
        with torch.no_grad():
            emb = self.model.get_embeddings(x,edge_index,edge_weights)
        return emb
    
    def _edge_prediction(self,x_emb1,x_emb2):
        assert self.best_state2 is not None, 'train classifier first'

        with torch.no_grad():
            out = self.model2(x_emb1,x_emb2)
        return out
        
    def train_step_flow(self, prior_esm_embed, prior_edge_index ,prior_edge_weights
                        ):
        self.model3.train()
        self.optimizer3.zero_grad()
        
        #keep = prior_edge_weights >= threshold
        u, v = prior_edge_index
        with autocast(self.device):
            with torch.no_grad():
                h = self.model.get_embeddings(prior_esm_embed,prior_edge_index,
                                              prior_edge_weights)
            
            loss  = torch.sum(self.model3.log_prob(self.cat_emb(h[u] ,h[v])))
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model3.parameters(), max_norm=1.0)
        self.optimizer3.step()
            
        return loss/len(u)
    
    def valid_step_flow(self, valid_esm_embed, valid_edge_index,valid_edge_weights):
        
        #keep = valid_edge_weights >= threshold
        u, v = valid_edge_index
        with torch.no_grad():
            h = self.model.get_embeddings(valid_esm_embed,valid_edge_index, valid_edge_weights)    
        
        self.model3.eval()    
        loss  = torch.sum(self.model3.log_prob(self.cat_emb(h[u] ,h[v])))
            
        return loss/len(u)
    
    def train_flow(self,
                    esm_embed, prior_edge_index ,prior_edge_weights,
                    valid_edge_index,valid_edge_weights,
                    percentile = 0.5, epochs = 100,verbose=True):
        assert self.best_state is not None, 'train embeddings first'
        
        self.threshold = torch.quantile(prior_edge_weights, percentile)
        prior_edge_index = prior_edge_index[:, prior_edge_weights >= self.threshold]
        prior_edge_weights = prior_edge_weights[prior_edge_weights >= self.threshold]
        valid_edge_index = valid_edge_index[:, valid_edge_weights >= self.threshold]
        valid_edge_weights = valid_edge_weights[valid_edge_weights >= self.threshold]        
        
        assert len(prior_edge_weights) >= 10, f'threshold too high, training set too small now, currently {len(prior_edge_weights)}'
        assert len(valid_edge_weights) >= 10, f'threshold too high, validation set too small now, currently {len(valid_edge_weights)}'
        
        if verbose:
            pbar = tqdm.tqdm(range(1,epochs+1),
                                desc="Training Progress", unit="epoch")
            for epoch in pbar:
                train_loss = self.train_step_flow(esm_embed, prior_edge_index ,prior_edge_weights)
                self.train_NLL_history.append(train_loss.item())
                valid_loss = self.valid_step_flow(esm_embed, valid_edge_index,valid_edge_weights)
                self.valid_NLL_history.append(valid_loss.item())
                if self.best_NLL > valid_loss:
                    self.best_NLL = valid_loss
                    self.best_state3 = {
                        k: v.cpu().clone()
                        for k, v in self.model3.state_dict().items()
                    }
                pbar.set_postfix({"Epoch": f"{epoch:4d}/{epochs}",
                                        "Tr": f"{train_loss:.4f}",
                                        "Vall": f"{valid_loss:.4f}"})
        else:
            for epoch in range(1,epochs+1):
                train_loss = self.train_step_flow(esm_embed, prior_edge_index ,prior_edge_weights)
                self.train_NLL_history.append(train_loss.item())
                valid_loss = self.valid_step_flow(esm_embed, valid_edge_index,valid_edge_weights)
                self.valid_NLL_history.append(valid_loss.item())
                if self.best_NLL > valid_loss:
                    self.best_NLL = valid_loss
                    self.best_state3 = {
                        k: v.cpu().clone()
                        for k, v in self.model3.state_dict().items()
                    }
                    
        if self.best_state3 is not None:
            self.model3.load_state_dict(self.best_state3)
        return self.best_NLL
    
    def _get_edge_prob(self,x_emb1, x_emb2):
        assert self.best_state3 is not None, 'train flow model first'
        emb = self.cat_emb(x_emb1,x_emb2)
        prob = self.model3._get_prob(emb)
        return prob
    
    def save(self, model: str, path: str):
        if model == "GAT":
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "train_losses": self.train_CL_history,
                    "val_losses": self.valid_CL_history,
                },
                path,
            )
        
        elif model == "MLP" :
            torch.save(
                {
                    "model_state": self.model2.state_dict(),
                    "optimizer_state": self.optimizer2.state_dict(),
                    "train_losses": self.train_MSE_history,
                    "val_losses": self.valid_MSE_history,
                },
                path,
            )
            
        elif model == "RealNVP" :
            torch.save(
                {
                    "model_state": self.model3.state_dict(),
                    "optimizer_state": self.optimizer3.state_dict(),
                    "train_losses": self.train_NLL_history,
                    "val_losses": self.valid_NLL_history,
                },
                path,
            )
        print(f"Saved checkpoint {model} → {path}")
    
    def load(self, model:str, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if model =="GAT":
            self.model.load_state_dict(ckpt["model_state"])
            self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.train_CL_history = ckpt.get("train_losses", [])
            self.valid_CL_history = ckpt.get("val_losses", [])
        elif model =="MLP":
            self.model2.load_state_dict(ckpt["model_state"])
            self.best_state2 = {
                    k: v.cpu().clone()
                    for k, v in self.model2.state_dict().items()
                }
            self.optimizer2.load_state_dict(ckpt["optimizer_state"])
            self.train_MSE_history = ckpt.get("train_losses", [])
            self.valid_MSE_history = ckpt.get("val_losses", [])
        elif model =="RealNVP":
            self.model3.load_state_dict(ckpt["model_state"])
            self.best_state3 = {
                    k: v.cpu().clone()
                    for k, v in self.model3.state_dict().items()
                }
            self.optimizer3.load_state_dict(ckpt["optimizer_state"])
            self.train_NLL_history = ckpt.get("train_losses", [])
            self.valid_NLL_history = ckpt.get("val_losses", [])
        print(f"Loaded checkpoint ← {path}")

