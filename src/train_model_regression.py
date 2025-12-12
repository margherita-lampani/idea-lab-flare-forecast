import sys,os
sys.path.append(os.getcwd())

#ADDED (disable wandb for testing on some systems)
os.environ["WANDB_MODE"] = "disabled"

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model_regressor import convnet_sc_regressor,LitConvNetRegressor
from data import MagnetogramDataModule
from data_zarr import AIAHMIDataModule
from src.mlp_model_regressor import MLPModel
from utils.model_utils import *
import pandas as pd
from pathlib import Path
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import yaml

def main():    
    # read in config file
    with open('experiment_config.yml') as config_file:
        config = yaml.safe_load(config_file.read())
    
    if config['meta']['resume']:
        run = wandb.init(config=config,project=config['meta']['project'],resume='must',id=config['meta']['id'])
    else:
        run = wandb.init(config=config,project=config['meta']['project'])
    config = wandb.config

    # set seeds
    pl.seed_everything(42,workers=True)

    #
    print('Features:',config.data['feature_cols'])

    # define data module
    if not config.data['use_zarr_dataset']:
        data = MagnetogramDataModule(data_file=config.data['data_file'],
                                    label=config.data['label'],
                                    balance_ratio=config.data['balance_ratio'],
                                    regression=config.data['regression'],
                                    val_split=config.data['val_split'],
                                    forecast_window=config.data['forecast_window'],
                                    dim=config.data['dim'],
                                    batch=config.training['batch_size'],
                                    augmentation=config.data['augmentation'],
                                    flare_thresh=config.data['flare_thresh'],
                                    flux_thresh=config.data['flux_thresh'],
                                    feature_cols=config.data['feature_cols'],
                                    test=config.data['test'],
                                    maxval=config.data['maxval'],
                                    file_col=config.data['file_col'])
    else:
        data = AIAHMIDataModule(zarr_file=config.data['zarr_file'],
                            val_split=config.data['val_split'],
                            data_file=config.data['data_file'],
                            regression=config.data['regression'],
                            forecast_window=config.data['forecast_window'],
                            dim=config.data['dim'],
                            batch=config.training['batch_size'],
                            augmentation=config.data['augmentation'],
                            flare_thresh=config.data['flare_thresh'],
                            feature_cols=config.data['feature_cols'],
                            test=config.data['test'],
                            channels=config.data['channels'],
                            maxvals=config.data['maxval'],)

    # train MLP model to obtain weights for final layers of CNN+MLP
    if len(config.data['feature_cols'])>0:
        mlp_model = MLPModel(data_file=config.data['data_file'],
                            window=config.data['forecast_window'],
                            val_split=config.data['val_split'],
                            flare_thresh=config.data['flare_thresh'],
                            features=config.data['feature_cols'],
                            hidden_layer_sizes=(100,))
        mlp_model.prepare_data()
        mlp_model.setup()
        mlp_model.train()
        weights = [mlp_model.model.coefs_[0],mlp_model.model.intercepts_[0],
                   mlp_model.model.coefs_[1],mlp_model.model.intercepts_[1]]
    else:
        weights = []
    
    # initialize model
    model = convnet_sc_regressor(dim=config.data['dim'],length=len(config.data['channels']),
                                 len_features=len(config.data['feature_cols']),
                                 weights=weights,dropoutRatio=config.model['dropout_ratio'])
    classifier = LitConvNetRegressor(model,config.training['lr'],config.training['wd'],epochs=config.training['epochs'])

    # load checkpoint (not needed, disabled for now)
    if wandb.run.resumed:
        classifier = load_model(run, config.meta['user']+'/'+config.meta['project']+'/model-'+config.meta['id']+':latest',
                                model, litclass=LitConvNetRegressor)
    elif config.model['load_checkpoint']:
        classifier = load_model(run, config.model['checkpoint_location'], model, 
                                litclass=LitConvNetRegressor, strict=False)


    # initialize wandb logger
    wandb_logger = WandbLogger(log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          mode='min',
                                          save_top_k=1,
                                          save_last=True,
                                          save_weights_only=True,
                                          verbose=False)
    early_stop_callback = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=20,mode='min',strict=False,check_finite=False) #patience=15

    # train model
    trainer = pl.Trainer(accelerator=config.training['device'],
                         #devices=[0],
                         devices=config.training.get('devices', 1), #ADDED
                         deterministic=False,
                         max_epochs=config.training['epochs'],
                         callbacks=[ModelSummary(max_depth=2),checkpoint_callback, early_stop_callback],
                         #gradient_clip_val=1.0, #ADDED
                         #logger=wandb_logger,
                         logger = True, 
                         enable_progress_bar=True, #ADDED
                         precision=16)
    
    # ============================================================
    # DEBUG 
    # ============================================================
    print("\n" + "="*60)
    print("DEBUG: Checking data BEFORE training")
    print("="*60)

    print("from my laptop")

    # Setup data
    data.prepare_data()
    data.setup('fit')

    # ============================================================
    # 1) NORMALIZED LABELS CHECK
    # ============================================================
    print("\n--- NORMALIZED LABELS CHECK ---")
    if hasattr(data, "train_set") and hasattr(data.train_set, "df"):
        df = data.train_set.df
        print("Found train_set.df")
        print(df['flare'].describe())
        print("Any NaN:", df['flare'].isna().any())
    else:
        print("WARNING: train_set.df not accessible — dataset does not store raw dataframe")


    # ============================================================
    # 2) TRAINING BATCH CHECK
    # ============================================================
    print("\n--- TRAINING BATCH CHECK ---")

    def summarize_tensor(name, t):
        """Summarize a tensor's properties."""
        print(f"\n{name}:")
        print(f"  Shape: {tuple(t.shape)}")
        print(f"  Dtype: {t.dtype}")
        try:
            print(f"  Min/Max: {t.min().item():.6f} / {t.max().item():.6f}")
            print(f"  Mean/Std: {t.mean().item():.6f} / {t.std().item():.6f}")
        except:
            print("  Could not compute stats (maybe empty?)")
        print(f"  Any NaN? {torch.isnan(t).any().item()}")
        print(f"  Any Inf? {torch.isinf(t).any().item()}")

    try:
        train_loader = data.train_dataloader()
        batch = next(iter(train_loader))

        print(f"\nBatch type: {type(batch)}")
        if isinstance(batch, tuple):
            print(f"Batch length: {len(batch)}")
        else:
            print("Batch is not a tuple — unexpected structure!")

        # ------------------------------------------------------------
        # Identify and summarize elements in the batch
        # ------------------------------------------------------------
        images = labels = features = None

        for i, element in enumerate(batch):
            print(f"\n--- Element {i} ---")
            print(f"Type: {type(element)}")

            # Case 1: tensor
            if torch.is_tensor(element):
                if element.ndim == 4:  # Batch di immagini (B, C, H, W)
                    images = element
                    summarize_tensor("Images", element)
                elif element.ndim == 1:  # probabilmente labels
                    labels = element
                    summarize_tensor("Labels", element)
                elif element.ndim == 2:  # features tabulari
                    features = element
                    summarize_tensor("Features", element)
                else:
                    print(f"Tensor of unexpected shape: {element.shape}")

            # Case 2: numpy array
            elif isinstance(element, np.ndarray):
                print("Numpy array detected, converting to tensor")
                t = torch.tensor(element)
                summarize_tensor(f"Element {i} (numpy)", t)

            # Case 3: nested tuple (multiple images)
            elif isinstance(element, tuple):
                print(f"Nested tuple (len={len(element)})")
                for j, sub in enumerate(element):
                    print(f"   sub-element {j}, type={type(sub)}")
                    if torch.is_tensor(sub):
                        summarize_tensor(f"Tuple[{i}][{j}]", sub)
            
            else:
                print(f"Unrecognized element type: {type(element)}")

    except Exception as e:
        print(f"ERROR loading batch: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("END DEBUG - Starting training...")
    print("="*60 + "\n")

    # ============================================================
    # END DEBUG 
    # ============================================================

    trainer.fit(model=classifier, datamodule=data)

    #TEMPORARILY DISABLED 
    # test trained model
    if config.testing['eval']:
        # load best checkpoint
        classifier = load_model(run, config.meta['user']+'/'+config.meta['project']+'/model-'+run.id+':best_k', model,
                                litclass=LitConvNetRegressor)

        # save predictions locally
        print('------Train/val predictions------')
        preds = trainer.predict(model=classifier,dataloaders=data.trainval_dataloader())
        save_preds(preds,wandb.run.dir,'trainval_results.csv',config.data['regression'])

        print('------Pseudotest predictions------')
        preds = trainer.predict(model=classifier,dataloaders=data.pseudotest_dataloader())
        save_preds(preds,wandb.run.dir,'pseudotest_results.csv',config.data['regression'])

        print('------Test predictions------')
        preds = trainer.predict(model=classifier,dataloaders=data.test_dataloader())
        save_preds(preds,wandb.run.dir,'test_results.csv',config.data['regression'])

    wandb.finish()
    
    # ============================================================
    # TESTING no wandb
    # ============================================================

    def save_predictions_local(preds, savedir, filename):
        
        rows = []
        for batch in preds:
            # From predict_step: fname, y_true, y_pred
            fnames, y_true, y_pred = batch

            fnames = list(fnames)
            y_true = y_true.detach().cpu().float().numpy().flatten().tolist()
            y_pred = y_pred.detach().cpu().float().numpy().flatten().tolist()

            for f, yt, yp in zip(fnames, y_true, y_pred):
                ae = abs(yt - yp)
                se = (yt - yp)**2
                
                row = {
                    "file": f,
                    "y_true": yt,
                    "y_pred": yp,
                    "AE": ae,
                    "SE": se,
                    "R2": None
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)

        # Compute overall metrics
        y_true_all = df["y_true"].astype(float).values
        y_pred_all = df["y_pred"].astype(float).values

        mae_mean = df["AE"].mean()
        mse_mean = df["SE"].mean()
        
        mae_median = df["AE"].median()
        mse_median = df["SE"].median()
        
        r2_overall = 1 - (
            np.sum((y_true_all - y_pred_all)**2) /
            np.sum((y_true_all - np.mean(y_true_all))**2)
        )

        # Summary
        df_summary = pd.DataFrame([
            {"file": "--- MEAN ---", "y_true": "", "y_pred": "", 
            "AE": mae_mean, "SE": mse_mean, "R2": r2_overall},
            {"file": "--- MEDIAN ---", "y_true": "", "y_pred": "", 
            "AE": mae_median, "SE": mse_median, "R2": ""},
        ])

        df_final = pd.concat([df, df_summary], ignore_index=True)

        os.makedirs(savedir, exist_ok=True)
        save_path = os.path.join(savedir, filename)
        df_final.to_csv(save_path, index=False)
        
        print(f"[DEBUG] Saved to: {save_path}")
        print(f"\nMetrics summary for {filename}:")
        print(f"  MAE (mean):   {mae_mean:.6f}")
        print(f"  MAE (median): {mae_median:.6f}")
        print(f"  MSE (mean):   {mse_mean:.6f}")
        print(f"  R2 (overall): {r2_overall:.6f}")

        return df_final
    
    if config.testing['eval']:
        print("\n" + "="*60)
        print("TESTING MODEL")
        print("="*60 + "\n")

        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"Loading best checkpoint from: {best_checkpoint_path}")
        
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            classifier = LitConvNetRegressor.load_from_checkpoint(
                best_checkpoint_path,
                model=model
            )
        else:
            print("WARNING: No checkpoint found, using current model")

        results_dir = config.testing['savedir']
        os.makedirs(results_dir, exist_ok=True)
        
        # --------------------------
        # Train/val predictions
        # --------------------------
        print('\n------Train/val predictions------')
        preds = trainer.predict(model=classifier, dataloaders=data.trainval_dataloader())
        save_predictions_local(preds, results_dir, 'trainval_results.csv')
        
        # --------------------------
        # Pseudotest predictions
        # --------------------------
        print('\n------Pseudotest predictions------')
        preds = trainer.predict(model=classifier, dataloaders=data.pseudotest_dataloader())
        save_predictions_local(preds, results_dir, 'pseudotest_results.csv')
        
        # --------------------------
        # Test predictions
        # --------------------------
        print('\n------Test predictions------')
        preds = trainer.predict(model=classifier, dataloaders=data.test_dataloader())
        save_predictions_local(preds, results_dir, 'test_results.csv')
        
        print("\nTesting complete! Results saved to:", results_dir)


    pass 

if __name__ == "__main__":
    main()
