import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy

class InterLabelLoss(nn.Module):

    def __init__(
        self,
        device:str="cuda",
    ):
        super(InterLabelLoss, self).__init__()

        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = device
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, y_pred, y_true, weight_tensor=None, aspect=None, print_loss=False, beta=1.0):
        """
        Compute the inter-label loss.

        Args:
            y_pred (_type_): predictions, no sigmoid
            y_true (_type_): ground truth
            weight_tensor (_type_, optional): GO term's information contect for weighting. Defaults to None.
            aspect (_type_, optional): Aspect of the model. Defaults to None.
            print_loss (bool, optional): Whether to print loss. Defaults to False.
            beta (float, optional): beta value for f1_score. Defaults to 1.0.
        """
        sig_y_pred = torch.sigmoid(y_pred)

        if weight_tensor is None:
            weight_tensor = torch.ones(y_pred.shape[1]).to(y_pred.device)
    
        crossentropy_loss = self.multilabel_categorical_crossentropy(y_pred, y_true)
        go_term_centric_loss = self.weight_go_centric_f1_loss(sig_y_pred, y_true, weight_tensor, mean='before', beta=beta)
        protein_centric_loss = self.weight_protein_centric_f1_loss(sig_y_pred, y_true, weight_tensor, mean='before', beta=beta)
        total_loss = crossentropy_loss * protein_centric_loss * go_term_centric_loss
        #total_loss = go_term_centric_loss * protein_centric_loss
        if print_loss:
            print('Crossentropy loss: {:.4f}, GO term loss: {:.4f}, Protein loss: {:.4f}, Total loss: {:.4f}'.format(crossentropy_loss.item(), go_term_centric_loss.item(), protein_centric_loss.item(), total_loss.item()))
        #crossentropy_loss1 = self.bce_loss(sig_y_pred, y_true)
        # if print_loss:
        #     print('Crossentropy loss: {:.4f}'.format(crossentropy_loss.item()))
        #total_loss =  crossentropy_loss + crossentropy_loss1 #* go_term_centric_loss

        return total_loss

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        
        # Modify predicted probabilities based on true labels
        y_pred = (1 - 2 * y_true) * y_pred
        
        # Adjust predicted probabilities
        y_pred_neg = y_pred - y_true * 1e16
        y_pred_pos = y_pred - (1 - y_true) * 1e16
        
        # Concatenate zeros tensor
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        
        # Compute logsumexp along the class dimension (dim=1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        total_loss = neg_loss + pos_loss
        
        return torch.mean(total_loss)
    
    def weight_protein_centric_f1_loss(self, y_pred, y_true, weight_tensor=None, mean:str="before", beta=1.0):

        if weight_tensor is None:
            weight_tensor = torch.ones(y_pred.shape[1]).to(y_pred.device)

        tp = torch.sum(y_true * y_pred * weight_tensor, dim=1).to(y_pred.device)
        fp = torch.sum((1 - y_true) * y_pred * weight_tensor, dim=1).to(y_pred.device)
        fn = torch.sum(y_true * (1 - y_pred) * weight_tensor, dim=1).to(y_pred.device)
        precision = tp / (tp + fp + 1e-16)
        recall = tp / (tp + fn + 1e-16)

        if mean == "before":
            mean_precision = torch.mean(precision)
            mean_recall = torch.mean(recall)
            f1 = self.f1_score( mean_precision, mean_recall, beta=beta)
            f1_loss = 1 - f1

        if mean == "after":
            f1 = self.f1_score( precision, recall, beta=beta)
            f1 = f1.mean()
            f1_loss = 1 - f1
        
        if mean == "none":
            f1 = self.f1_score( precision, recall, beta=beta)
            f1_loss = 1 - f1

        return f1_loss 

    def weight_go_centric_f1_loss(self, y_pred, y_true, weight_tensor=None, mean:str="after", beta=1.0):

        if weight_tensor is None:
            weight_tensor = torch.ones(y_pred.shape[1]).to(y_pred.device)

        tp = torch.sum(y_true * y_pred * weight_tensor, dim=0).to(y_pred.device)
        fp = torch.sum((1 - y_true) * y_pred * weight_tensor, dim=0).to(y_pred.device)
        fn = torch.sum(y_true * (1 - y_pred) * weight_tensor, dim=0).to(y_pred.device)

        precision = tp / (tp + fp + 1e-16)
        recall = tp / (tp + fn + 1e-16)

        if mean == "before":
            mean_precision = torch.mean(precision)
            mean_recall = torch.mean(recall)
            f1 = self.f1_score( mean_precision, mean_recall, beta=beta)
            return 1 - f1

        if mean == "after":
            f1 = self.f1_score( precision, recall, beta=beta)
            f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
            f1 = f1.mean()
            return 1 - f1
        
        if mean == 'none':
            f1 = self.f1_score( precision, recall, beta=beta)
            f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
            return 1 - f1
        
    def f1_score(self, precision, recall, beta=0.5, eps=1e-16):
        f1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + eps)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return f1
    

class EarlyStop:
    """
    Early stopping class.
    """
    def __init__(self, patience:int=5, min_epochs:int=50, monitor:str='loss'):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_f1 = None
        self.early_stop = False
        self.f1_backup_model = None
        self.loss_backup_model = None
        self.min_epochs = min_epochs
        self.monitor = monitor
        self.epoch = 0
    
    def __call__(self, loss, f1_score, model):
        self.epoch += 1
        if self.best_f1 is None or self.best_loss is None:
            self.best_f1 = f1_score
            self.best_loss = loss
            self.backup_model = deepcopy(model.state_dict())

        loss_improved = False
        f1_improved = False

        if loss < self.best_loss or self.epoch < self.min_epochs:
            self.best_loss = loss
            self.loss_backup_model = deepcopy(model.state_dict())
            loss_improved = True
        
        if f1_score > self.best_f1 or self.epoch < self.min_epochs:
            self.best_f1 = f1_score
            self.f1_backup_model = deepcopy(model.state_dict())
            f1_improved = True

        
        if self.monitor == 'loss':
            if not loss_improved:
                self.counter += 1
                print(f'Early stop counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0
        
        elif self.monitor == 'f1_score':
            if not f1_improved:
                self.counter += 1
                print(f'Early stop counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0
        
        elif self.monitor == 'both':
            if not loss_improved and not f1_improved:
                self.counter += 1
                print(f'Early stop counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0

    def stop(self):
        return self.early_stop
    
    def restore(self, model):
        if self.monitor == 'loss':
            model.load_state_dict(self.loss_backup_model)
        elif self.monitor == 'f1_score':
            model.load_state_dict(self.f1_backup_model)
        elif self.monitor == 'both':
            model.load_state_dict(self.f1_backup_model)
        return model

    def has_backup_model(self):
        if self.monitor == 'loss':
            return self.loss_backup_model is not None
        elif self.monitor == 'f1_score':
            return self.f1_backup_model is not None
        elif self.monitor == 'both':
            return self.f1_backup_model is not None


class FmaxMetric:

    def __init__(self,
        weight_matrix:np.ndarray=None,
        child_matrix:np.ndarray=None,
        device:str="cuda",
    ):
        if torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'
        
        if isinstance(weight_matrix, np.ndarray):
            self.weight_matrix = torch.from_numpy(weight_matrix).float().to(self.device)
        elif isinstance(weight_matrix, torch.Tensor):
            self.weight_matrix = weight_matrix.to(self.device)
        elif weight_matrix is None:
            self.weight_matrix = None
        else:
            raise ValueError(f'weight_matrix must be numpy.ndarray or torch.Tensor, not {type(weight_matrix)}')

        if isinstance(child_matrix, np.ndarray):
            self.child_matrix = torch.from_numpy(child_matrix).float().to(self.device)
        elif isinstance(child_matrix, torch.Tensor):
            self.child_matrix = child_matrix.to(self.device)
        elif child_matrix is None:
            self.child_matrix = None
        else:
            raise ValueError(f'child_matrix must be numpy.ndarray or torch.Tensor, not {type(child_matrix)}')
        
    def compute_protein_centric_fm(self, y_pred, y_true, margin=0.01, weight_tensor=None, parent_prop=False):
        # if parent_prop and self.child_matrix is not None:
        #     y_pred = self.parent_prop(y_pred, self.child_matrix)

        if weight_tensor is None:
            if self.weight_tensor is None:
                weight_tensor = torch.ones_like(y_pred)
            else:
                weight_tensor = self.weight_tensor
        
        cut_off_array = np.arange(0, 1, margin)
        # create a tensor to store the f1 score for each cutoff, shape is (100, f1)
        record_f1_tensor = torch.zeros(len(cut_off_array),dtype=torch.float).to(y_pred.device)
        # gruond truth
        n_gt = y_true.sum(dim=1).to(y_pred.device)
        wn_gt = torch.sum(weight_tensor * y_true, dim=1).to(y_pred.device)
        other_record_tensor = torch.zeros((len(cut_off_array), 3),dtype=torch.float).to(y_pred.device)

        for i, threshold in enumerate(cut_off_array):

            solidified_y_pred = self.solidify_prediction(y_pred, threshold)
            # number of proteins with at least one term predicted with score >= tau
            cov = torch.sum(torch.sum(solidified_y_pred, dim=1) > 0).to(y_pred.device)
            # Subsets size
            intersection = torch.logical_and(solidified_y_pred, y_true)
            n_pred = solidified_y_pred.sum(dim=1).to(y_pred.device)
            n_intersection = intersection.sum(dim=1).to(y_pred.device)

            wn_pred = torch.sum(weight_tensor * solidified_y_pred, dim=1).to(y_pred.device)
            wn_intersection = torch.sum(weight_tensor * intersection, dim=1).to(y_pred.device)

            w_precision = torch.where(n_pred > 0, torch.div(wn_intersection.float(), wn_pred.float()), torch.zeros_like(n_intersection)).sum()
            w_recall = torch.where(n_gt > 0, torch.div(wn_intersection.float(), wn_gt.float()), torch.zeros_like(n_intersection)).sum()

            w_precision = w_precision / cov
            w_recall = w_recall / cov

            n = 2 * w_precision * w_recall
            d = w_precision + w_recall
            w_f1 = n / (d + 1e-16)
            record_f1_tensor[i] = w_f1
            other_record_tensor[i, 0] = w_precision
            other_record_tensor[i, 1] = w_recall
            other_record_tensor[i, 2] = threshold
        
        # find the best f1 score and return the f1 score, precision, recall and threshold
        best_f1 = record_f1_tensor.max()
        best_f1_idx = record_f1_tensor.argmax()
        return best_f1, * other_record_tensor[best_f1_idx, :]

    def solidify_prediction(self, y_pred:torch.tensor, threshold=0.5)->torch.tensor:
        """
        Solidify the prediction.
        If the prediction is larger than the threshold, set it to 1.
        If the prediction is smaller than the threshold, set it to 0.

        Args:
            y_pred (torch.tensor): prediction
            threshold (float, optional): threshold for prediction. Defaults to 0.5.

        Returns:
            torch.tensor: solidified prediction
        """
        solidified_y_pred = torch.zeros_like(y_pred).to(y_pred.device)
        solidified_y_pred[y_pred > threshold] = 1
        return solidified_y_pred
    
    def parent_prop(self, y_pred, child_matrix, batch_size:int=32)->torch.tensor:
        """
        use the child_matrix to update child value to parent value if the child value is larger than the parent value.

        Args:
            y_pred (torch.tensor): prediction
            child_matrix (torch.tensor): child matrix where child_matrix[i][j] = 1 if the jth GO term is a subclass of the ith GO term else 0
            batch_size (int, optional): batch size for parent prop, if num of class is large, please lower the batch size for memory consideration. Defaults to 32.

        Returns:
            torch.tensor: updated prediction
        """ 
        org_shape = y_pred.shape
        batch_size = min(batch_size, y_pred.shape[0])
        child_matrix_unsq = child_matrix.unsqueeze(0).to(y_pred.device)
        for i in range(0, y_pred.shape[0], batch_size):
            cur_y_pred = y_pred[i:i+batch_size]
            cur_y_pred = torch.max(cur_y_pred.unsqueeze(1) * child_matrix_unsq, dim = -1)[0]
            y_pred[i:i+batch_size] = cur_y_pred
        return y_pred.reshape(org_shape)        


class Trainer:

    def __init__(self,
        model:torch.nn.Module,
        train_loader:torch.utils.data.DataLoader,
        val_loader:torch.utils.data.DataLoader=None,
        learning_rate:float=0.001,
        epochs:int=5000,
        weight_matrix:torch.tensor=None,
        child_matrix:torch.tensor=None,
        device:str="cuda",
        optimizer:torch.optim.Optimizer=None,
        log_interval:int=1,
        eval_interval:int=1,
        loss_fn:torch.nn.Module=None,
        scheduler:torch.optim.lr_scheduler.ReduceLROnPlateau=None,
        metric:FmaxMetric=None,
        patience:int=5,
        min_epochs:int=50,
        early_stopping:bool=True,
        aspect:str='BPO',
        parent_prop:bool=False, # whether to use parent_prop to update child value to parent value for evaluation
        monitor:str='loss',
    ) -> None:
        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.history = {
            'train' : dict(),
            'val' : dict()
        }
        self.lr = learning_rate

        if not early_stopping:
            min_epochs = epochs
        self.monitor = monitor
        self.early_stopping = early_stopping
        self.early_stop = EarlyStop(patience=patience, min_epochs=min_epochs, monitor=monitor)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3, amsgrad=True, betas=(0.9, 0.999), eps=1e-6)
        self.optimizer = optimizer
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)
        self.scheduler = scheduler

        if isinstance(weight_matrix, np.ndarray):
            self.weight_matrix = torch.from_numpy(weight_matrix).float().to(self.device)
        elif isinstance(weight_matrix, torch.Tensor):
            self.weight_matrix = weight_matrix.to(self.device)
        elif weight_matrix is None:
            self.weight_matrix = None
        else:
            raise ValueError(f'weight_matrix must be numpy.ndarray or torch.Tensor, not {type(weight_matrix)}')

        if isinstance(child_matrix, np.ndarray):
            self.child_matrix = torch.from_numpy(child_matrix).float().to(self.device)
        elif isinstance(child_matrix, torch.Tensor):
            self.child_matrix = child_matrix.to(self.device)
        elif child_matrix is None:
            self.child_matrix = None
        else:
            raise ValueError(f'child_matrix must be numpy.ndarray or torch.Tensor, not {type(child_matrix)}')
        
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.aspect = aspect
        self.epochs = epochs
        self.parent_prop = parent_prop

        if loss_fn is None:
            loss_fn = InterLabelLoss(device=self.device)
        self.loss_fn = loss_fn

        if metric is None:
            metric = FmaxMetric(device=self.device, weight_matrix=weight_matrix, child_matrix=child_matrix)
        self.metric = metric

    def single_epoch(self, model=None, train_loader=None):
        """train the model for one epoch

        Args:
            model (nn.Module, optional): DNN model. Defaults to None.
            dataset (torch.utils.data.Dataset, optional): training dataset. Defaults to None.
            batch_size (int, optional): batch size. Defaults to None.
        """
        if model is None:
            model = self.model
        if train_loader is None:
            train_loader = self.train_loader
        model.train()
        loss_sum = 0.0
     
        for batch_idx, (names, inputs, y_true) in tqdm(enumerate(train_loader), total=len(train_loader), ascii=' >='):
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            y_true = y_true.to(self.device)
            y_pred = model(inputs)
            loss = self.loss_fn(y_pred, y_true, weight_tensor=self.weight_matrix, aspect=self.aspect, print_loss=False)
            loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()

        mean_loss = loss_sum / len(train_loader)
        return mean_loss

    def fit(self, model=None, train_loader=None, val_loader=None, epochs=None):
        if model is None:
            model = self.model
        if train_loader is None:
            train_loader = self.train_loader
        if val_loader is None:
            val_loader = self.val_loader
        if epochs is None:
            epochs = self.epochs

        pre_best_loss, mean_f1_score, mean_precision, mean_recall, cut_off = np.inf, 0, 0, 0, 0
        self.early_stop.best_f1 = mean_f1_score
        self.early_stop.best_loss = pre_best_loss
        print(f'pre_best_loss: {pre_best_loss}, fmax: {mean_f1_score}')
        
        for epoch_idx in range(epochs):
            mean_loss = self.single_epoch(model=model, train_loader=self.train_loader)
            
            # log the training information
            if epoch_idx % self.log_interval == 0:
                self.history['train'][epoch_idx] = {
                    'loss' : mean_loss,
                }
                print(f'Train Epoch {epoch_idx}: loss: {mean_loss}')
    

            mean_val_loss, mean_f1_score, mean_precision, mean_recall, cut_off = self.evaluate(model=model, val_loader=val_loader, margin=0.01, print_loss=True)

            if epoch_idx % self.eval_interval == 0:
                # logging
                self.history['val'][epoch_idx] = {
                    'loss' : mean_val_loss,
                    'f1_score' : mean_f1_score,
                    'precision' : mean_precision,
                    'recall' : mean_recall,
                    'cut_off' : cut_off
                }
                print(f'Val Epoch {epoch_idx}: loss: {mean_val_loss}, fmax: {mean_f1_score}, p: {mean_precision}, r: {mean_recall}, t: {cut_off}')

            # early stop monitor
            self.early_stop(mean_val_loss, mean_f1_score, model)

            self.scheduler.step(mean_val_loss)
            if self.early_stop.stop():
                if self.early_stop.has_backup_model():
                    model = self.early_stop.restore(model)
                    save_model = True
                else:
                    save_model = False
                print(f'Early stop at epoch {epoch_idx}, best loss: {self.early_stop.best_loss}, current loss: {mean_val_loss}')
                print(f'best f1: {self.early_stop.best_f1}, current f1: {mean_f1_score}')
                return save_model, self.early_stop.best_loss, self.early_stop.best_f1, self.early_stop.epoch

        if self.early_stopping:
            model = self.early_stop.restore(model)
        return True, self.early_stop.best_loss, self.early_stop.best_f1, self.early_stop.epoch

    def evaluate(self, model=None, val_loader=None, margin=0.001, print_loss=False):
        if model is None:
            model = self.model
        if val_loader is None:
            val_loader = self.val_loader

        model.eval()
        with torch.no_grad():
            y_preds = []
            y_trues = []
            total_loss = 0
            for batch_idx, (names, inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                y_pred = model(inputs)
                loss = self.loss_fn(y_pred, labels, weight_tensor=self.weight_matrix, print_loss=print_loss, aspect=self.aspect)
                y_pred = torch.sigmoid(y_pred) # apply sigmoid because the model's output is logits
                y_preds.append(y_pred)
                y_trues.append(labels)
                total_loss += loss.item()
            
            loss = total_loss / len(val_loader)
            y_preds = torch.cat(y_preds, dim=0)
            y_trues = torch.cat(y_trues, dim=0)
            
            mean_f1_score, mean_precision, mean_recall, cut_off = self.metric.compute_protein_centric_fm(y_preds, y_trues, margin=margin, weight_tensor=self.weight_matrix, parent_prop=self.parent_prop)
            mean_f1_score = mean_f1_score.item()
            mean_precision = mean_precision.item()
            mean_recall = mean_recall.item()
            cut_off = cut_off.item()
            # round the cut_off to 3 decimal places
            cut_off = round(cut_off, 3)
            # round other to 5 decimal places
            mean_f1_score = round(mean_f1_score, 5)
            mean_precision = round(mean_precision, 5)
            mean_recall = round(mean_recall, 5)
            return loss, mean_f1_score, mean_precision, mean_recall, cut_off


class Predictor:

    def __init__(self,
        model: nn.Module=None,
        PredictLoader:torch.utils.data.DataLoader=None,
        device:str='cuda', 
        child_matrix:torch.tensor=None,
        back_prop:bool=False, # whether to use parent_prop to update child value to parent value for prediction
    ) -> None:
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = device

        if model is not None:
            self.model = model
            self.model.eval()
            self.model.prediction_mode = True
        else:
            self.model = None

        if PredictLoader is not None:
            self.PredictLoader = PredictLoader
        else:
            self.PredictLoader = None

        self.back_prop = back_prop

        if isinstance(child_matrix, np.ndarray):
            self.child_matrix = torch.from_numpy(child_matrix).float().to(self.device)
        elif isinstance(child_matrix, torch.Tensor):
            self.child_matrix = child_matrix.to(self.device)
        elif child_matrix is None:
            self.child_matrix = None
        else:
            raise ValueError(f'child_matrix must be numpy.ndarray or torch.Tensor, not {type(child_matrix)}')

    def parent_prop(self, y_pred, child_matrix, batch_size:int=32)->torch.tensor:
        """
        use the child_matrix to update child value to parent value if the child value is larger than the parent value.

        Args:
            y_pred (torch.tensor): prediction
            child_matrix (torch.tensor): child matrix where child_matrix[i][j] = 1 if the jth GO term is a subclass of the ith GO term else 0
            batch_size (int, optional): batch size for parent prop, if num of class is large, please lower the batch size for memory consideration. Defaults to 32.

        Returns:
            torch.tensor: updated prediction
        """ 
        org_shape = y_pred.shape
        batch_size = min(batch_size, y_pred.shape[0])
        child_matrix_unsq = child_matrix.unsqueeze(0).to(y_pred.device)
        for i in range(0, y_pred.shape[0], batch_size):
            cur_y_pred = y_pred[i:i+batch_size]
            cur_y_pred = torch.max(cur_y_pred.unsqueeze(1) * child_matrix_unsq, dim = -1)[0]
            y_pred[i:i+batch_size] = cur_y_pred
        return y_pred.reshape(org_shape)     
    
    def update_loader(self, PredictLoader):
        self.PredictLoader = PredictLoader
    
    def update_model(self, model, child_matrix):
        self.model = model
        if isinstance(child_matrix, np.ndarray):
            self.child_matrix = torch.from_numpy(child_matrix).float().to(self.device)
        elif isinstance(child_matrix, torch.Tensor):
            self.child_matrix = child_matrix.to(self.device)
        elif child_matrix is None:
            self.child_matrix = None
        else:
            raise ValueError(f'child_matrix must be numpy.ndarray or torch.Tensor, not {type(child_matrix)}')
        self.model.eval()
        self.model.prediction_mode = True

    def predict(self):
        self.model.eval()
        self.model.prediction_mode = True
        with torch.no_grad():
            y_preds = []
            protein_ids = []
            for i, (names, features) in tqdm(enumerate(self.PredictLoader),total=len(self.PredictLoader), ascii=' >=', desc='Predicting'):
                protein_ids.extend(names)
                features = features.to(self.device)
                y_pred = self.model(features)
                if self.back_prop and self.child_matrix is not None:
                    y_pred = self.parent_prop(y_pred, self.child_matrix)
                y_pred = y_pred.cpu().numpy()
                y_preds.append(y_pred)
            
            y_preds = np.concatenate(y_preds, axis=0)
        return protein_ids, y_preds


