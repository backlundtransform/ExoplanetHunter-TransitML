import pandas as pd


class DataPreprocessor:
    @staticmethod
    def preprocess_data(data, frequency ='D', datetime_column ='DateTime', label_column = 'Label'):
        data_copy =  data.copy()
        data_copy[datetime_column] = pd.to_datetime(data_copy[datetime_column])
        data_copy.set_index(datetime_column, inplace=True)
        groups =   data_copy.index.to_period(frequency).astype(str)

        unique_labels_per_group = data_copy.groupby(groups)[label_column].nunique()
        inconsistent_groups = unique_labels_per_group[unique_labels_per_group > 1]
    
        if not inconsistent_groups.empty:
          print(f"Warning: {len(inconsistent_groups)} groups have more than one unique label, which may affect the model's performance.")
    
        X = data.drop(columns=[label_column,  datetime_column])
        y = data[label_column]
    
        return X, y, groups