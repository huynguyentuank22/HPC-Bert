import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import random
import time

class MachineLearning:
    def __init__(self,X_train,X_val,X_test,Y_train,Y_val,Y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_test = Y_test

        self.configs = {
            "mlp": {
                # MLPRegressor mặc định hidden_layer_sizes=(100,), activation='relu', solver='adam'
                # Thêm batch_size để huấn luyện nhanh hơn trên dataset lớn
                "batch_size": 128,
                "max_iter": 200,  # mặc định 200 (giữ nguyên)
            },
            "randomforest": {
                # RandomForestRegressor: chỉ thêm n_jobs để tận dụng đa nhân CPU
                "n_jobs": -1,
            },
            "xgboost": {
                # XGBRegressor: chỉ thêm n_jobs để huấn luyện song song
                "n_jobs": -1,
            },
            "gradientboosting": {
                # GradientBoostingRegressor không có n_jobs (không hỗ trợ song song)
                # => giữ mặc định hoàn toàn
            },
            "extratrees": {
                # ExtraTreesRegressor: thêm n_jobs để tận dụng đa nhân CPU
                "n_jobs": -1,
            },
            "lasso": {
                # Lasso: thêm max_iter để tránh dừng sớm, giữ mặc định alpha=1.0
                "max_iter": 1000,
            }
            # "mlp": {"hidden_layer_sizes": (64,128,64), "activation": "relu", "solver": "adam", "max_iter": 1000, "learning_rate_init": 0.001},
            # "randomforest": {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, 'max_features': None, 'bootstrap': True, "n_jobs": -1},
            # "xgboost": {"n_estimators": 50, "learning_rate": 0.2, "max_depth": 3, "verbosity": 1, "n_jobs": -1},
            # "gradientboosting": {"loss": "squared_error", "learning_rate": 0.2, "n_estimators": 50, "subsample": 0.8},
            # "extratrees": {"n_estimators": 50, "max_depth": 10, "min_samples_split": 5, "random_state": 42, "n_jobs": -1},
            # "lasso": {"alpha": 10,"fit_intercept": True,"max_iter": 500,"tol": 0.001,"random_state": 42},
            # "knn": {"n_neighbors": 3, "weights": "uniform", "algorithm": "auto", "leaf_size": 30, "p": 2},
            # "linearreg": {"fit_intercept": False, "copy_X": True, "n_jobs": -1},
            # "decisiontree": {"criterion": "squared_error", "max_depth": 20, "min_samples_split": 3},
            # "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": -1, "num_leaves": 31, "random_state": 42},
            # "catboost": {"iterations": 100, "learning_rate": 0.1, "depth": 6, "verbose": False},
            # "svr": {"C": 1,"epsilon": 0.1,"kernel": "linear"},
            # "elasticnet": {"alpha": 1,"l1_ratio": 0.5,"fit_intercept": True,"max_iter": 1000,"tol":0.001},
            # "ridge": {"alpha": 0.1,"fit_intercept": True,'solver': 'auto'}
        }
        
        self.param_grids = {
            "mlp": {
                "hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 128)],
                "activation": ["relu", "tanh", "logistic"],
                "solver": ["adam", "sgd"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"],
                "max_iter": [200, 300, 400]
            },
            "randomforest": {
                "n_estimators": [100, 200, 300, 500],
                "criterion": ["squared_error", "absolute_error", "poisson"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
            },
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0]
            },
            "gradientboosting": {
                "loss": ["squared_error", "huber", "absolute_error", "quantile"],
                "learning_rate": [0.01, 0.1, 0.2],
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "subsample": [0.6, 0.8, 1.0],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "extratrees": {
                "n_estimators": [100, 200, 300],
                "criterion": ["squared_error", "absolute_error"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
            },
            "lasso": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
                "max_iter": [500, 1000, 5000],
                "tol": [1e-4, 1e-3, 1e-2]
            },
            # "knn": {
            #     "n_neighbors": [3, 5, 7, 9],
            #     "weights": ["uniform", "distance"],
            #     "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            #     "leaf_size": [10,30,50,100],
            #     "p": [1, 2]  
            # },
            # "linearreg": {
            #     "fit_intercept": [True, False],
            #     "copy_X": [True, False]
            # },
            # "decisiontree": {
            #     "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            #     "splitter": ["best", "random"],
            #     "max_depth": [None, 10, 20, 30, 40, 50],
            #     "min_samples_split": [2, 5, 10],
            #     "min_samples_leaf": [1, 2, 4]
            # },
            # "lightgbm": {
            #     "n_estimators": [100, 200, 300],
            #     "learning_rate": [0.01, 0.1, 0.2],
            #     "max_depth": [-1, 5, 10, 15],
            #     "num_leaves": [31, 50, 100],
            #     "subsample": [0.6, 0.8, 1.0],
            #     "colsample_bytree": [0.6, 0.8, 1.0]
            # },
            # "catboost": {
            #     "iterations": [100, 200, 300],
            #     "learning_rate": [0.01, 0.1, 0.2],
            #     "depth": [4, 6, 8, 10],
            #     "l2_leaf_reg": [1, 3, 5, 7],
            #     "border_count": [32, 64, 128]
            # },
            # "svr": {
            #     "C": [0.1, 1, 10, 100],
            #     "epsilon": [0.01, 0.1, 0.2, 0.5],
            #     "kernel": ["linear", "rbf", "poly"],
            #     "degree": [2, 3, 4], 
            #     "gamma": ["scale", "auto"]
            # },
            # "elasticnet": {
            #     "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            #     "l1_ratio": [0.1, 0.5, 0.9],
            #     "fit_intercept": [True, False],
            #     "max_iter": [500, 1000, 5000],
            #     "tol": [1e-4, 1e-3, 1e-2]
            # },
            # "ridge": {
            #     "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            #     "fit_intercept": [True, False],
            #     "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "lbfgs"]
            # },
        }
      
    def create_model(self, model_name):
        if model_name not in self.configs:
            raise ValueError(f"Model '{model_name}' không được hỗ trợ.")
        try:
            model_class = {
                "mlp": MLPRegressor,
                "randomforest": RandomForestRegressor,
                "xgboost": XGBRegressor,
                "gradientboosting": GradientBoostingRegressor,
                "extratrees": ExtraTreesRegressor,
                "lasso": Lasso,

                # "knn": KNeighborsRegressor,
                # "linearreg": LinearRegression,
                # "decisiontree": DecisionTreeRegressor,
                # "lightgbm": LGBMRegressor,
                # "catboost": CatBoostRegressor,
                # "svr": SVR,
                # "elasticnet": ElasticNet,
                # "ridge": Ridge,
            }[model_name]
            return model_class(**self.configs[model_name])
        except KeyError:
            raise ValueError(f"Model '{model_name}' không tồn tại.")


    def update_config(self, model_name, new_config):
        if model_name in self.configs:
            self.configs[model_name] = {**self.configs[model_name], **new_config}
        return self.create_model(model_name)

    
    def scatter_plot(self, model_name):
        model = self.create_model(model_name)
        model.fit(self.X_train, self.Y_train)
        y_pred = model.predict(self.X_test)
        plt.figure(figsize=(6, 4))
        plt.scatter(self.Y_test, y_pred, alpha=0.5)
        plt.plot([self.Y_test.min(), self.Y_test.max()], [self.Y_test.min(), self.Y_test.max()], 'r--')
        plt.title(f"Actual vs Predicted for {model_name}", fontsize=16)
        plt.xlabel("Actual Values", fontsize=14)
        plt.ylabel("Predicted Values", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def line_plot(self,model_name):
        window = 750
        model = self.create_model(model_name)
        model.fit(self.X_train, self.Y_train)
        y_pred = model.predict(self.X_test)
        y_true_smooth = gaussian_filter1d(self.Y_test, sigma=window/5)
        y_pred_smooth = gaussian_filter1d(y_pred, sigma=window/5)
    
        plt.figure(figsize=(12, 6))  
        
        plt.plot(y_true_smooth, label="Actual Value", color="dodgerblue", linewidth=2)
        plt.plot(y_pred_smooth, label="Predicted Value", color="orangered", linestyle="--", linewidth=2)
        
        plt.xlabel("Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        
        plt.legend(fontsize=12, loc="best")
        
        plt.tight_layout()
        plt.show()


    def run_model(self,model_name,kind = ''):
        model = self.create_model(model_name)
        model.fit(self.X_train, self.Y_train)
        y_pred = model.predict(self.X_test)
        print(f"\nPerformance for {model_name}:")
        print(f"Config: {self.configs[model_name]}")
        print(f"Mean Squared Error: {mean_squared_error(self.Y_test, y_pred):.4f}")
        print(f"Mean Absolute Error: {mean_absolute_error(self.Y_test, y_pred):.4f}")
        print(f"R² Score: {r2_score(self.Y_test, y_pred):.4f}")
        if kind == 'line':
            self.line_plot(model_name)
        elif kind == 'scatter':
            self.scatter_plot(model_name)


    def random_search_for_single_model(self, model_name, cv=5, scoring='r2', verbose=1, n_iter=30, random_state=42):
        if model_name not in self.configs:
                raise ValueError(f"Model '{model_name}' not recognized. Available models: {list(self.configs.keys())}")
    
        model = self.create_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in the class instance.")
    
        param_distributions = self.param_grids.get(model_name, None)
        if param_distributions is None:
            raise ValueError(f"No parameter grid defined for model '{model_name}'.")
    
        randomized_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=verbose,
            random_state=random_state
        )
    
        print(f"Starting Randomized Search for model '{model_name}'...")
        randomized_search.fit(self.X_train, self.Y_train)
    
        best_params = randomized_search.best_params_
        best_score = randomized_search.best_score_
    
        print(f"Best Parameters for {model_name}: {best_params}")
        print(f"Best Cross-Validation {scoring}: {best_score:.4f}")

        return self.update_config(model_name, best_params)

    
    def compare_models(self, models):
        performance_data = []

        for model_name in models:
            model = self.create_model(model_name)
            if model is None:
                print(f"Model '{model_name}' not found. Skipping.")
                continue
            try:
                model.fit(self.X_train, self.Y_train)
                y_pred = model.predict(self.X_test)

                metrics = {
                    "Model": model_name.upper(),
                    "RMSE": np.sqrt(mean_squared_error(self.Y_test, y_pred)),
                    "MAE": mean_absolute_error(self.Y_test, y_pred),
                    "R² Score": r2_score(self.Y_test, y_pred)
                }
                performance_data.append(metrics)
            except Exception as e:
                print(f"Error evaluating model '{model_name}': {e}")
                continue
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.sort_values(by="R² Score", ascending=False, inplace=True)
        return performance_data

    def get_results(self, models):
        performance_data = []

        for model_name in models:
            model = self.create_model(model_name)
            if model is None:
                print(f"Model '{model_name}' not found. Skipping.")
                continue
            try:
                model.fit(self.X_train, self.Y_train)
                y_pred = model.predict(self.X_test)

                metrics = {
                    'result':y_pred,
                    "Model": model_name.upper()
                }
                performance_data.append(metrics)
            except Exception as e:
                print(f"Error evaluating model '{model_name}': {e}")
                continue
        
        return performance_data
    
class Dynamic_Weighted_Ensemble:
    def __init__(self,X_train,X_val,X_test,Y_train,Y_val,Y_test,scaler=None):
        self.ml = MachineLearning(X_train,X_val,X_test,Y_train,Y_val,Y_test)
        self.scaler = scaler
        self.num_sub = 3
        
        # Init value for optimize
        self.lr = 0.1
        self.discount_factor = 0.6
        self.window_size = 200
        self.drift_threshold = 0.1

        # Parameter grid for optimization
        self.param_grid = {
                    'lr': [0.01, 0.05, 0.1, 0.2, 0.5],
                    'discount_factor': [0.3, 0.5, 0.6, 0.8, 1.0],
                    'window_size': [100, 200, 500, 1000],
                    'drift_threshold': [0.05, 0.1, 0.2, 0.3]
                }
        
        # Init model
        self.models = {}
        self.weights = {}

        # Init metrics
        self.performance_metrics = {}
        self.pre_performance_metrics = {}

        # Init prediction
        self.prediction = {}
        self.validation = {}
        self.prediction_test = {}

        # Meta Model
        self.meta_model = None
        self.meta_model_name = ''

    
    def init_base_sub(self, model_list: list[list[str]]):
        for i in range(1,self.num_sub + 1):
            sub_ensemble_key = f"sub_ensemble{i}"
            if sub_ensemble_key not in self.models:
                self.models[sub_ensemble_key] = {}
            if sub_ensemble_key not in self.weights:
                self.weights[sub_ensemble_key] = {}
            if sub_ensemble_key not in self.prediction:
                self.prediction[sub_ensemble_key] = {}
            if sub_ensemble_key not in self.validation:
                self.validation[sub_ensemble_key] = {}
            if sub_ensemble_key not in self.prediction_test:
                self.prediction_test[sub_ensemble_key] = {}   

            for model_name in model_list[i - 1]:
                # model = getattr(self.ml, model_name, None)
                model = self.ml.create_model(model_name)
                self.models[sub_ensemble_key][model_name] = model
                self.weights[sub_ensemble_key][model_name] = 1/ len(model_list[i - 1])
    
    def train_base_model(self, sub_ensemble_name, name, model):
        # print('Training ',sub_ensemble_name, ' Model ',name)
        model.fit(self.ml.X_train, self.ml.Y_train)
        preds = model.predict(self.ml.X_train)
        self.prediction[sub_ensemble_name][name] = preds
        

    def training_subensemble(self):    
        for i in range(1,self.num_sub + 1):
            sub_ensemble_key = f"sub_ensemble{i}"
            for name, model in self.models[sub_ensemble_key].items():
                self.train_base_model(sub_ensemble_key, name, model)

    def valid_subensemble(self, X_window, y_window):
        for i in range(1, self.num_sub + 1):  
            sub_ensemble_key = f"sub_ensemble{i}"
    
            if sub_ensemble_key not in self.validation:
                self.validation[sub_ensemble_key] = {}
    
            if sub_ensemble_key not in self.performance_metrics:
                self.performance_metrics[sub_ensemble_key] = {}
    
            for name, model in self.models[sub_ensemble_key].items():
                self.validation[sub_ensemble_key][name] = model.predict(X_window)
    
                mae = mean_absolute_error(y_window, self.validation[sub_ensemble_key][name])
                rmse = np.sqrt(mean_squared_error(y_window, self.validation[sub_ensemble_key][name]))
                r2 = r2_score(y_window, self.validation[sub_ensemble_key][name])
    
                self.performance_metrics[sub_ensemble_key][name] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                }
                
    def update_weights(self):
        for i in range(1, self.num_sub + 1):  
            sub_ensemble_key = f"sub_ensemble{i}"
    
            if sub_ensemble_key not in self.weights:
                self.weights[sub_ensemble_key] = {}
    
            if sub_ensemble_key not in self.pre_performance_metrics:
                self.pre_performance_metrics[sub_ensemble_key] = {}
    
            for name, metrics in self.performance_metrics[sub_ensemble_key].items():
                performance_metric = metrics["R2"]
                pre_performance_metric = self.pre_performance_metrics[sub_ensemble_key].get(name, {}).get("R2", 0)
    
                self.weights[sub_ensemble_key][name] = self.weights[sub_ensemble_key].get(name, 0) + \
                    self.lr * (pre_performance_metric - performance_metric) * self.discount_factor
    
            total_weight = sum(self.weights[sub_ensemble_key].values())
            if total_weight > 0:  
                for name in self.weights[sub_ensemble_key]:
                    self.weights[sub_ensemble_key][name] /= total_weight
            
            self.pre_performance_metrics[sub_ensemble_key] = self.performance_metrics[sub_ensemble_key].copy()


    def soft_regression(self, predictions, weights):
        final_prediction = sum(predictions[name] * weights[name] for name in predictions.keys())
        return final_prediction / sum(weights.values()) 

    def sliding_window_training(self):
        total_samples = len(self.ml.X_val)
    
        for start in range(0, total_samples - self.window_size + 1, self.window_size):
            end = start + self.window_size
            X_window = self.ml.X_val[start:end]
            y_window = self.ml.Y_val[start:end]
    
            # Validate tất cả sub-ensembles
            self.valid_subensemble(X_window, y_window)
    
            pred_subensembles = []
            for i in range(1, self.num_sub + 1):
                sub_ensemble_key = f"sub_ensemble{i}"
                pred = self.soft_regression(self.validation[sub_ensemble_key], self.weights[sub_ensemble_key])
                pred_subensembles.append(pred)
    
            meta_X_valid = np.column_stack(pred_subensembles)
            final_predict = self.meta_model.predict(meta_X_valid)
    
            final_mae = mean_absolute_error(y_window, final_predict)
            final_rmse = np.sqrt(mean_squared_error(y_window, final_predict))
            final_r2 = r2_score(y_window, final_predict)
    
            self.final_performance_metrics = {
                "MAE": final_mae,
                "RMSE": final_rmse,
                "R2": final_r2
            }

            if hasattr(self, 'pre_final_performance_metrics') and self.pre_final_performance_metrics['R2'] > self.final_performance_metrics['R2'] + self.drift_threshold:

                self.update_weights()
    
                # for i in range(1, self.num_sub + 1):
                #     sub_ensemble_key = f"sub_ensemble{i}"
                #     print(f"Updated weights for {sub_ensemble_key}:")
                #     for model_name, weight in self.weights[sub_ensemble_key].items():
                #         print(f"  - {model_name}: {weight:.4f}")
    
            self.pre_final_performance_metrics = self.final_performance_metrics.copy()

    def line_plot(self,y_pred,y_test):
        window = 750
        y_true_smooth = gaussian_filter1d(y_test, sigma=window/5)
        y_pred_smooth = gaussian_filter1d(y_pred, sigma=window/5)
    
        plt.figure(figsize=(12, 6))  
        
        plt.plot(y_true_smooth, label="Actual Value", color="dodgerblue", linewidth=2)
        plt.plot(y_pred_smooth, label="Predicted Value", color="orangered", linestyle="--", linewidth=2)
        
        plt.xlabel("Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        
        plt.legend(fontsize=12, loc="best")
        
        plt.tight_layout()
        plt.show()


    def model_predict(self, X_test, y_test):
        start_time = time.time()
        for i in range(1, self.num_sub + 1):
            sub_ensemble_key = f"sub_ensemble{i}"
            if sub_ensemble_key not in self.prediction_test:
                self.prediction_test[sub_ensemble_key] = {}
    
            for name, model in self.models[sub_ensemble_key].items():
                self.prediction_test[sub_ensemble_key][name] = model.predict(X_test)
    
        pred_subensembles = []
        for i in range(1, self.num_sub + 1):
            sub_ensemble_key = f"sub_ensemble{i}"
            pred = self.soft_regression(self.prediction_test[sub_ensemble_key], self.weights[sub_ensemble_key])
            pred_subensembles.append(pred)
    
        meta_X_test = np.column_stack(pred_subensembles)
        final_pred = self.meta_model.predict(meta_X_test)

        # Inverse transform predictions and actual values
        # Reshape for inverse transform
        final_pred_reshaped = final_pred.reshape(-1, 1)
        y_test_reshaped = y_test.reshape(-1, 1)
        
        # Create dummy arrays with same number of columns as during scaling
        dummy_array_pred = np.zeros((final_pred_reshaped.shape[0], self.scaler.n_features_in_))
        dummy_array_test = np.zeros((y_test_reshaped.shape[0], self.scaler.n_features_in_))
        
        # Assuming target_column is the last column in the scaled data
        target_idx = -1  # Adjust if target_column is not the last column
        dummy_array_pred[:, target_idx] = final_pred_reshaped.flatten()
        dummy_array_test[:, target_idx] = y_test_reshaped.flatten()
        
        # Inverse transform
        final_pred = self.scaler.inverse_transform(dummy_array_pred)[:, target_idx]
        y_test = self.scaler.inverse_transform(dummy_array_test)[:, target_idx]
        end_time = time.time()
        
        # self.line_plot(final_pred,y_test)
        final_mae = mean_absolute_error(y_test, final_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
        final_r2 = r2_score(y_test, final_pred)
        name = ''
        if self.num_sub == 1:
            name = 'Normal Stacking'
        else :
            name = 'HDEP'
        self.metrics = {
            'result':final_pred,
            "Model": name,
            "MAE": final_mae,
            "RMSE": final_rmse,
            "R² Score": final_r2,
            "Inference Time": (end_time - start_time)/len(final_pred)
        }

        print('Predict on Test Set')
        print("MAE: ", final_mae)
        print("RMSE: ", final_rmse)
        print("R2: ", final_r2)
        print('Inference Time: ', (end_time - start_time)/len(final_pred))
        return self.metrics


        

    def run_model(self,grid = False):
        self.training_subensemble() 
    
        pred_subensembles = []
        for i in range(1, self.num_sub + 1):
            sub_ensemble_key = f"sub_ensemble{i}"
            pred = self.soft_regression(self.prediction[sub_ensemble_key], self.weights[sub_ensemble_key])
            pred_subensembles.append(pred)
    
        meta_X_train = np.column_stack(pred_subensembles)
        #-------------------------------------------------------------------------------------------
        if grid == True: 
            param_grids = self.ml.param_grids.get(self.meta_model_name,{})
            model = self.ml.create_model(self.meta_model_name)
            self.meta_model = RandomizedSearchCV(model, param_distributions=param_grids, n_iter=30, cv=5, scoring='r2',verbose=1, random_state=42, n_jobs=-1)
            self.meta_model.fit(meta_X_train,self.ml.Y_train)
            print('Best Param ',self.meta_model.best_estimator_)
            print('Best Score ',self.meta_model.best_score_)
        #------------------------------------------------------------------------
        # self.meta_model =  GradientBoostingRegressor(loss='absolute_error', max_depth=7,
        #                   min_samples_split=10, n_estimators=200,
        #                   subsample=0.6)
        #--------------------------------------------------------------------------------
        else:
            self.meta_model = self.ml.create_model(self.meta_model_name)
            self.meta_model.fit(meta_X_train,self.ml.Y_train)
        #--------------------------------------------------------------------------------

        self.sliding_window_training()
        print('Key on Test Set')
        return self.model_predict(self.ml.X_test, self.ml.Y_test)
    
    def random_search_params(self, model_list, meta_model_name, n_iter=20, scoring='r2', random_state=42):
        """
        Perform random search over HDEM parameters (lr, discount_factor, window_size, drift_threshold).
        
        Parameters:
        - model_list: list[list[str]], list of model combinations for sub-ensembles
        - meta_model_name: str, name of the meta-model
        - n_iter: int, number of parameter combinations to try
        - scoring: str, metric to optimize ('r2', 'mae', or 'rmse')
        - random_state: int, random seed for reproducibility
        
        Returns:
        - best_params: dict, best parameters found
        - best_score: float, best score achieved
        """
        # Initialize best parameters and score
        best_score = -float('inf') if scoring == 'r2' else float('inf')
        best_params = {}
        
        # Generate random combinations of parameters
        param_combinations = []
        for _ in range(n_iter):
            params = {
                'lr': random.choice(self.param_grid['lr']),
                'discount_factor': random.choice(self.param_grid['discount_factor']),
                'window_size': random.choice(self.param_grid['window_size']),
                'drift_threshold': random.choice(self.param_grid['drift_threshold'])
            }
            param_combinations.append(params)
        
        print(f"Starting Random Search for HDEM parameters ({n_iter} iterations)...")
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\nIteration {i}/{n_iter}: {params}")
            
            # Set parameters
            self.lr = params['lr']
            self.discount_factor = params['discount_factor']
            self.window_size = params['window_size']
            self.drift_threshold = params['drift_threshold']
            self.meta_model_name = meta_model_name
            
            # Initialize and train model
            self.init_base_sub(model_list)
            self.training_subensemble()
            
            pred_subensembles = []
            for j in range(1, self.num_sub + 1):
                sub_ensemble_key = f"sub_ensemble{j}"
                pred = self.soft_regression(self.prediction[sub_ensemble_key], self.weights[sub_ensemble_key])
                pred_subensembles.append(pred)
            
            meta_X_train = np.column_stack(pred_subensembles)
            self.meta_model = self.ml.create_model(meta_model_name)
            self.meta_model.fit(meta_X_train, self.ml.Y_train)
            
            # Evaluate on validation set
            self.sliding_window_training()
            
            # Get validation score
            score = self.final_performance_metrics[scoring.upper()]
            if scoring != 'r2':
                score = -score  # Minimize MAE and RMSE
            
            print(f"Validation {scoring.upper()}: {score:.4f}")
            
            # Update best parameters if score improves
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"New best parameters: {best_params}, Score: {best_score:.4f}")
        
        # Set the best parameters
        self.lr = best_params.get('lr', self.lr)
        self.discount_factor = best_params.get('discount_factor', self.discount_factor)
        self.window_size = best_params.get('window_size', self.window_size)
        self.drift_threshold = best_params.get('drift_threshold', self.drift_threshold)
        
        print(f"\nBest Parameters: {best_params}")
        print(f"Best Validation {scoring.upper()}: {best_score:.4f}")

        return best_params, best_score

    def random_search_params_test(self, model_list, meta_model_name, n_iter=20, scoring='r2', random_state=42):
        """
        Perform random search over HDEM (parameters: lr, discount_factor, window_size, drift_threshold).
        
        Parameters:
        - model_list: list[list[str]], list of model combinations for sub-ensembles
        - meta_model_name: str, name of the meta-model
        - n_iter: int, number of parameter combinations to try
        - scoring: str, metric to optimize ('r2', 'mae', or 'rmse')
        - random_state: int, random seed for reproducibility
        
        Returns:
        - best_params: dict, best parameters found
        - best_score: float, best validation score achieved
        - results: list of dicts, all results with parameters
        """
        # Initialize best parameters and score
        best_score = -float('inf') if scoring == 'r2' else float('inf')
        best_params = {}
        results = []
        
        # Generate random combinations of parameters
        param_combinations = []
        for _ in range(n_iter):
            params = {
                'lr': random.choice(self.param_grid['lr']),
                'discount_factor': random.choice(self.param_grid['discount_factor']),
                'window_size': random.choice(self.param_grid['window_size']),
                'drift_threshold': random.choice(self.param_grid['drift_threshold'])
            }
            param_combinations.append(params)
        
        print(f"Starting Random Search for HDEM parameters ({n_iter} iterations)...")
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\nIteration {i}/{n_iter}: {params}")
            
            # Set parameters
            self.lr = params['lr']
            self.discount_factor = params['discount_factor']
            self.window_size = params['window_size']
            self.drift_threshold = params['drift_threshold']
            self.meta_model_name = meta_model_name
            
            # Initialize and train model
            self.init_base_sub(model_list)
            self.training_subensemble()
            
            pred_subensembles = []
            for j in range(1, self.num_sub + 1):
                sub_ensemble_key = f"sub_ensemble{j}"
                pred = self.soft_regression(self.prediction[sub_ensemble_key], self.weights[sub_ensemble_key])
                pred_subensembles.append(pred)
            
            meta_X_train = np.column_stack(pred_subensembles)
            self.meta_model = self.ml.create_model(meta_model_name)
            self.meta_model.fit(meta_X_train, self.ml.Y_train)
            
            # Evaluate on validation set
            self.sliding_window_training()
            validation_score = self.final_performance_metrics[scoring.upper()]
            if scoring != 'r2':
                validation_score = -validation_score
            # Minimize MAE and RMSE
            
            # Evaluate on test set
            test_metrics = self.model_predict(self.ml.X_test, self.ml.Y_test)
            test_score = test_metrics['R² Score']
            
            print(f"Validation {scoring.upper()}: {validation_score:.4f}")
            print(f"Test R²: {test_score:.4f}")
            print(f"Test MAE: {test_metrics['MAE']:.4f}")
            print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
            
            # Store results
            result = {
                'params': params,
                'validation_score': validation_score,
                'test_r2': test_metrics['R² Score'],
                'test_mae': test_metrics['MAE'],
                'test_rmse': test_metrics['RMSE']
            }
            results.append(result)
            
            # Update best parameters if validation score improves
            if validation_score > best_score:
                best_score = validation_score
                best_params = params.copy()
                print(f"New best parameters: {best_params}, Validation Score: {best_score:.4f}")
        
        # Set the best parameters
        self.lr = best_params.get('lr', self.lr)
        self.discount_factor = best_params.get('discount_factor', self.discount_factor)
        self.window_size = best_params.get('window_size', self.window_size)
        self.drift_threshold = best_params.get('drift_threshold', self.drift_threshold)
        
        print(f"\nBest Parameters: {best_params}")
        print(f"Best Validation {scoring.upper()}: {best_score:.4f}")
        
        return best_params, best_score, results