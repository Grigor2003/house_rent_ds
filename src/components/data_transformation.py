import sys
import os
from dataclasses import dataclass
from itertools import product

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import HousingException
from src.logger import logging
from src.utils import save_object
from src.components.combined_onehot_encoder import CombinedOneHotEncoder
from src.config import (
    LOG_SCALING_COLS, 
    CATEGORICAL_COLS, 
    NUMERIC_COLS, 
    COMBINATIONS_TO_APPLY,
    PRETTY_NAMES,
    COLUMN_SHORT_NAMES,
)


def prettify_value(col_name: str, value) -> str:
    """Convert raw category value to pretty display name."""
    if col_name in PRETTY_NAMES:
        pretty = PRETTY_NAMES[col_name].get(value)
        if pretty:
            return pretty
    return str(value)


def prettify_col_name(col_name: str) -> str:
    """Convert column name to short display name."""
    return COLUMN_SHORT_NAMES.get(col_name, col_name)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    features_preprocessor_obj_file_path = os.path.join('artifacts', "features_preprocessor.pkl")
    combiner_obj_file_path = os.path.join('artifacts', "combiner.pkl")


class FeaturesGenerator(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self


    def transform(self, X):
        try:
            logging.info("Adding new features")           
            # X = X.dropna()
            X['Floor Level'] = X['Floor Level'].astype(int)
            X['Total Floors'] = X['Total Floors'].astype(int)

            X['Posted On'] = pd.to_datetime(X['Posted On'])
            # X['month posted'] = X['Posted On'].dt.month
            # X['day posted'] = X['Posted On'].dt.day
            X['day of week posted'] = X['Posted On'].dt.day_of_week
            X['quarter posted'] = X['Posted On'].dt.quarter

            X.drop('Posted On', axis = 1, inplace= True)
            logging.info("New features added")

            return X

        except Exception as e:
            raise HousingException(e, sys)



class LogScaling(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self   

    def transform(self, X):
        return np.log(X)


def process_floor_column(df):
    """Parse Floor column into Floor Level, drop original."""
    df = df.copy()
    
    split_floor = df['Floor'].str.split(' out of ', expand=True)
    df['Floor Level'] = split_floor[0]
    df['Total Floors'] = split_floor[1]
    
    df['Floor Level'] = df['Floor Level'].replace({
        'Ground': 0,
        'Lower Basement': -1,
        'Upper Basement': -2
    })
    df['Floor Level'] = pd.to_numeric(df['Floor Level'], errors='coerce')
    df['Total Floors'] = pd.to_numeric(df['Total Floors'], errors='coerce')
    
    df['Floor Level'] = df['Floor Level'].fillna(df['Floor Level'].median())
    df['Total Floors'] = df['Total Floors'].fillna(df['Total Floors'].median())
    
    df = df.drop(['Floor'], axis=1)
    return df


def get_onehot_groups(preprocessor, log_cols, num_cols, cat_cols):
    start_idx = len(log_cols) + len(num_cols)
    
    ohe = preprocessor.named_transformers_['cat_pipelines'].named_steps['one_hot_encoder']
    
    onehot_groups = {}
    for col_name, categories in zip(cat_cols, ohe.categories_):
        size = len(categories)
        onehot_groups[col_name] = {'start': start_idx, 'size': size}
        start_idx += size
    
    return onehot_groups


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            log_scaling_cols = LOG_SCALING_COLS
            cat_cols = CATEGORICAL_COLS
            num_cols = NUMERIC_COLS

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]
            )

            feature_eng_pipeline = Pipeline(
                steps=[
                    ('feature_generator', FeaturesGenerator())
                ]
            )

            preprocessor = ColumnTransformer([
                ("log_transform", LogScaling(), log_scaling_cols),
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipelines", cat_pipeline, cat_cols)
            ], remainder='passthrough')

            return feature_eng_pipeline, preprocessor, log_scaling_cols, num_cols, cat_cols
        
        except Exception as e:
            raise HousingException(e, sys)



    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data Read successfully")

            train_df = process_floor_column(train_df)
            test_df = process_floor_column(test_df) 


            logging.info("Obtaining preprocessing object")

            features_obj, preprocessing_obj, log_cols, num_cols, cat_cols = self.get_data_transformer_object()

            target_column_name = "Rent"

            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = np.log(train_df[target_column_name])

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = np.log(test_df[target_column_name])

            logging.info("Creating new Features")
            X_train = features_obj.fit_transform(X_train)
            X_test = features_obj.transform(X_test)

            logging.info(f"Applying preprocessing object on training and test set")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            logging.info("Saving preprocessing object")
            pd.DataFrame(X_train_arr).to_csv('artifacts/X_train_arr_preprocessing.csv', index=False)
            pd.DataFrame(X_test_arr).to_csv('artifacts/X_test_arr_preprocessing.csv', index=False)

            onehot_groups = get_onehot_groups(preprocessing_obj, log_cols, num_cols, cat_cols)
            logging.info(f"OneHot groups: {onehot_groups}")
            
            combinations_to_apply = COMBINATIONS_TO_APPLY
            
            # Validate: check that columns are not used after being removed
            # A column is removed when used with keep_originals=False
            # After removal, it cannot be used in any subsequent combination
            removed_columns = set()
            for combo_config in combinations_to_apply:
                keep_originals = combo_config.get('keep_originals', False)
                for col in combo_config['columns']:
                    if col in removed_columns:
                        error_msg = (
                            f"Column '{col}' was already removed by a previous combination with keep_originals=False. "
                            f"It cannot be used in subsequent combinations. "
                            f"Reorder combinations so that keep_originals=True comes first, "
                            f"or merge them into a single combination."
                        )
                        logging.error(error_msg)
                        raise ValueError(error_msg)
                # Mark columns as removed if keep_originals=False
                if not keep_originals:
                    for col in combo_config['columns']:
                        removed_columns.add(col)
            
            combiners = []
            combined_groups = {}  # Track combined groups and their positions
            
            for combo_config in combinations_to_apply:
                combo = combo_config['columns']
                keep_originals = combo_config.get('keep_originals', False)
                
                indices = []
                sizes = []
                for group_name in combo:
                    if group_name not in onehot_groups:
                        error_msg = (
                            f"Group '{group_name}' not found. "
                            f"Only non-repeating combinations can be used. "
                            f"Available groups: {list(onehot_groups.keys())}"
                        )
                        logging.error(error_msg)
                        raise ValueError(error_msg)
                    group = onehot_groups[group_name]
                    indices.extend(range(group['start'], group['start'] + group['size']))
                    sizes.append(group['size'])
                
                logging.info(f"Combining: {combo}, sizes={sizes}, indices={indices}, keep_originals={keep_originals}")
                combiner = CombinedOneHotEncoder(column_indices=indices, sizes=sizes, keep_originals=keep_originals)
                X_train_arr = combiner.fit_transform(X_train_arr)
                X_test_arr = combiner.transform(X_test_arr)
                combiners.append(combiner)
                
                # Track the combined group (added at the end)
                combined_size = np.prod(sizes)
                combined_start = X_train_arr.shape[1] - combined_size
                combined_name = ' x '.join(combo)
                combined_groups[combined_name] = {'start': combined_start, 'size': combined_size}
                
                if not keep_originals:
                    # Remove original groups from tracking (they no longer exist in data)
                    for group_name in combo:
                        del onehot_groups[group_name]

                    indices_set = set(indices)
                    # Recalculate indices: for each remaining group, count how many columns were removed before it
                    for name, group in onehot_groups.items():
                        removed_before = sum(1 for idx in indices_set if idx < group['start'])
                        group['start'] = group['start'] - removed_before
                    
                    # Also update previously combined groups
                    for name, group in combined_groups.items():
                        if name != combined_name:
                            removed_before = sum(1 for idx in indices_set if idx < group['start'])
                            group['start'] = group['start'] - removed_before
                
                logging.info(f"After combining {combo}: shape={X_train_arr.shape}, remaining groups: {list(onehot_groups.keys())}, keep_originals={keep_originals}")
            
            
            # Drop first column from EACH one-hot group to avoid dummy variable trap
            # This includes: remaining onehot_groups (Tenant Preferred) + combined_groups
            all_groups = {**onehot_groups, **combined_groups}
            columns_to_drop = [group['start'] for group in all_groups.values()]
            columns_to_drop = sorted(columns_to_drop, reverse=True)  # Drop from end to preserve indices
            
            for col_idx in columns_to_drop:
                X_train_arr = np.delete(X_train_arr, col_idx, axis=1)
                X_test_arr = np.delete(X_test_arr, col_idx, axis=1)
            
            logging.info(f"Dropped first column of each group {list(all_groups.keys())} to avoid dummy variable trap. New shape: {X_train_arr.shape}")
            
            # Generate feature names dynamically
            feature_names = []
            
            # Add log-scaled columns
            feature_names.extend([f'log({col})' for col in log_cols])
            
            # Add numeric columns
            feature_names.extend(num_cols)
            
            # Get OneHotEncoder categories for remaining groups (after combining)
            ohe = preprocessing_obj.named_transformers_['cat_pipelines'].named_steps['one_hot_encoder']
            
            # Remaining one-hot groups (not combined, or combined but with keep_originals=True)
            removed_cols = [item for combo_config in combinations_to_apply 
                           for item in combo_config['columns'] 
                           if not combo_config.get('keep_originals', False)]
            remaining_cat_cols = [c for c in cat_cols if c not in removed_cols]
            cat_col_to_categories = dict(zip(cat_cols, ohe.categories_))
            
            for col in remaining_cat_cols:
                categories = cat_col_to_categories[col]
                short_col = prettify_col_name(col)
                # Skip first category (dropped for dummy trap)
                for cat in categories[1:]:
                    pretty_val = prettify_value(col, cat)
                    feature_names.append(f'{short_col}: {pretty_val}')
            
            # Add combined group names
            for combo_config in combinations_to_apply:
                combo = combo_config['columns']
                combo_categories = [cat_col_to_categories[c] for c in combo]
                all_combos = list(product(*combo_categories))
                # Skip first (dropped for dummy trap)
                for combo_vals in all_combos[1:]:
                    parts = []
                    for c, v in zip(combo, combo_vals):
                        short_col = prettify_col_name(c)
                        pretty_val = prettify_value(c, v)
                        parts.append(f'{short_col}={pretty_val}')
                    feature_names.append(' & '.join(parts))
            
            logging.info(f"Generated {len(feature_names)} feature names")
            
            # Create DataFrame for easier column access
            X_train_df = pd.DataFrame(X_train_arr, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_arr, columns=feature_names)
            
            # Save with column names
            X_train_df.to_csv('artifacts/X_train_arr.csv', index=False)
            X_test_df.to_csv('artifacts/X_test_arr.csv', index=False)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            save_object(
                file_path = self.data_transformation_config.features_preprocessor_obj_file_path,
                obj = features_obj
            )

            save_object(
                file_path = self.data_transformation_config.combiner_obj_file_path,
                obj = combiners  # list of CombinedOneHotEncoder for each combination
            )

            logging.info(f"Saved preprocessing objects. Combiners count: {len(combiners)}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.features_preprocessor_obj_file_path,
                self.data_transformation_config.combiner_obj_file_path
            )
        except Exception as e:
            raise HousingException(e,sys)
