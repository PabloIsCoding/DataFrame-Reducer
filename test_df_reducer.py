import numpy as np
import pandas as pd
from df_reducer import (
    determine_type, 
    lower_upper, 
    optimal_int, 
    optimal_object, 
    reduce_size)

# Tests to be run with pytest

def test_determine_type():
    assert determine_type(pd.Series([1,2,3,4])) == 'int'
    assert determine_type(pd.Series([1.,2,3,4])) == 'float'
    assert determine_type(pd.to_datetime(['2019-01-01', '2019-01-02'])) == 'object'
    assert determine_type(pd.Series(['a', 'b'])) == 'object'
    
    

def test_lower_upper():
    s = pd.Series([1,2,3,4])
    assert lower_upper(s) == (0.8, 4.8)
    assert lower_upper(-s) == (-4.8, -0.8)
    assert lower_upper(-s, allow_negatives=True) == (-4.8, -0.8)
    assert lower_upper(-s, allow_negatives=False) == (-4.8, -0.8) # allow_negatives is ignored if the series isn't positive
    assert lower_upper(s, margin=5) == (0, 24)
    assert lower_upper(s, margin=5, allow_negatives=False) == (0, 24)
    assert lower_upper(s, margin=5, allow_negatives=True) == (-4, 24)

def test_optimal_int():
    """
    If the series is positive, optimal_int only returns 'int' if 
    allow_negatives=True explicitly. 
    """

    # positive
    positive = pd.Series([1, 2, 3])
    assert optimal_int(positive) == 'uint8'
    assert optimal_int(positive, margin=2) == 'uint8' # min_value would be -1, but allow_negatives=None, so min_value=0
    assert optimal_int(positive,
                       allow_negatives=False, margin=2) == 'uint8'
    assert optimal_int(positive,
                       allow_negatives = True, margin = 2) == 'int8' 

    # one_neg
    one_neg = pd.Series([-1, 2, 3])
    assert optimal_int(one_neg) == 'int8'
    assert optimal_int(one_neg, allow_negatives=True) == 'int8'
    assert optimal_int(one_neg, allow_negatives=False) == 'int8'

    # zero
    zero = pd.Series([0, 1, 2])
    assert optimal_int(zero) == 'uint8'  
    assert optimal_int(zero, allow_negatives=True) == 'uint8' # 0*margin=0

    # Number too big for int32, but okay in uint32
    s = pd.Series([2500e6])
    assert optimal_int(s) == 'uint32'
    assert optimal_int(s*2) == 'uint64'

    # Number okay in both uint32 and int32, returns uint
    assert optimal_int(pd.Series([2000e6])) == 'uint32'

def test_optimal_object():
    dates = pd.date_range('2016-02-29', '2018-07-15', freq='D')
    assert optimal_object(dates) == 'datetime'
    assert optimal_object(pd.Series(['a','b'])) == 'drop'
    assert optimal_object(pd.Series(['a','b','b','b','b'])) == 'category'

def test_reduce_size():
    df = pd.DataFrame({
        'a': list(range(1, 8)),
        'b': [1.1, 1.2, 3, 4, 5, 6, 7],
        'c': [1, np.nan, 3, 4, 5, 6, 7],
        'd': ['x', 'y', 'x', 'x', 'x', 'x', 'x'],
        'e': ['x', 'y', 'z', 'z', 'z', 'z', 'z'],
        'f': pd.date_range('2016-02-27', '2016-03-04', freq='D'),
        'g': [-1, -1, -2, -2, -1, -2, -1]
    })

    # dtypes1
    dtypes1 = pd.Series(['uint8', 'float16', 'float16', 'category', 
                          'category','datetime64[ns]', 'int8'],
                          df.columns)
    assert (reduce_size(df.copy()).dtypes == dtypes1).all()
    assert (reduce_size(df.copy(), margin=5,
                        allow_negatives=False).dtypes == dtypes1).all()

    # dtypes2
    dtypes2 = dtypes1.copy()
    dtypes2[0] = 'int8'
    assert (reduce_size(df.copy(), margin=5,
            allow_negatives=True).dtypes == dtypes2).all()

    # dtypes3
    dtypes3 = dtypes1.copy()
    dtypes3[1] = 'uint8'
    assert (reduce_size(df.copy(), 
            round_cols=['b']).dtypes == dtypes3).all()
    
    # dtypes4
    dtypes4 = dtypes1.copy()
    dtypes4[-1] = 'bool'
    assert (reduce_size(df.copy(), 
            int_to_bool=True).dtypes == dtypes4).all()

    
if __name__ == "__main__":
    test_lower_upper()
    test_optimal_int()
    test_reduce_size()
