import pandas as pd
import numpy as np
### Optimize Pandas Dataframe memory usage based on column value distribution.


def determine_type(series):
    # TODO add uint and maybe create more kinds of general types
    # object has a specific meaning in pandas, at least this should be changed
    if series.dtype in [np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64]:
        return "int"
    elif series.dtype in [np.float16, np.float32, np.float64]:
        return "float"
    else:
        return "object"
    
def lower_upper(series, margin=0.2, allow_negatives=None):
    """
    Returns an upper and lower bound for the series,
    based on its distribution.
    
    If margin is None, the upper and lower bounds are
    just the min and max of the series.
    
    If it's a number, it adds a margin below and above
    in proportion to the min and max, respectively.
    
    margin can be greater than 1 and also 0.
    """
    min_val = series.min()
    max_val = series.max()
    # If series is positive, by default it's assumed to be positive on new data, too:
    min_zero = min_val >= 0 and (allow_negatives is None or not allow_negatives)
    low = min_val - abs(min_val)*margin
    lower = max(0, low) if min_zero else low
    upper = max_val + abs(max_val)*margin
    
    return lower, upper
    
def optimal_int(series, margin=0.2, allow_negatives=None, int_to_bool=False):
    lower, upper = lower_upper(series, margin, allow_negatives)
    unsigned = lower >= 0
    final_type = 'int64'
        
    if int_to_bool and len(series.unique()) == 2:
        final_type = 'bool'
    elif unsigned:
        candidates = ['uint8', 'uint16', 'uint32']
        final_type = 'uint64'
        for candidate in candidates:
            if upper < np.iinfo(candidate).max:
                final_type = candidate
                break
    else: # When signed
        candidates = ['int8', 'int16', 'int32']
        for candidate in candidates:
            if lower > np.iinfo(candidate).min and \
               upper < np.iinfo(candidate).max:
                final_type = candidate
                break

    return final_type

def optimal_float(series, margin=0.2, allow_negatives=None):
    lower, upper = lower_upper(series, margin, allow_negatives)
    final_type = 'float64'
    if lower > -100 and upper < 100:
        final_type = 'float16'
    elif lower > -1e6 and upper < 1e6:
        final_type = 'float32'
    return final_type

def optimal_object(series):
    
    # Dates
    try:
        pd.to_datetime(series)
        return "datetime" # Pandas only allows this for datetime
    except:
        pass
    
    # Too many categories
    if len(series.unique()) >= len(series)/2:
        return "drop"
    
    # Few categories
    else:
        return "category"
    
    

def reduce_size(dataframe, 
                round_cols=None, 
                drop=False, 
                margin=0.2, 
                allow_negatives=None,
                int_to_bool=False):
    for col in dataframe:
        col_type = determine_type(dataframe[col])
        
        if col_type == 'int':
            optimal_type = optimal_int(dataframe[col], margin, allow_negatives, int_to_bool)
            dataframe[col] = dataframe[col].astype(optimal_type)

        elif col_type == 'float':
            # Round float col to make it int
            needs_rounding = round_cols is not None and col in round_cols
            if needs_rounding:
                try:
                    dataframe[col] = round(dataframe[col]).astype(int)
                    # Now it's int, so we optimize int:
                    optimal_type = optimal_int(dataframe[col], margin, allow_negatives)
                    dataframe[col] = dataframe[col].astype(optimal_type)
                except: # Maybe np.nan were found, so we must keep float but optimize it
                    # TODO: Revise this to see if we can avoid copying and pasting code from the else
                    optimal_type = optimal_float(dataframe[col], margin, allow_negatives)
                    dataframe[col] = dataframe[col].astype(optimal_type)
            else:
                optimal_type = optimal_float(dataframe[col], margin, allow_negatives)
                dataframe[col] = dataframe[col].astype(optimal_type)
        else:
            optimal_type = optimal_object(dataframe[col])
            if optimal_type == 'drop' and drop:
                dataframe.drop(col, axis=1, inplace=True)
            elif optimal_type == 'datetime':
                dataframe[col] = pd.to_datetime(dataframe[col])
            elif optimal_type != 'drop':
                dataframe[col] = dataframe[col].astype(optimal_type)
                
    return dataframe
        
    
