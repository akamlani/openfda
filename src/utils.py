import numpy as np
from pprint import pprint
import numericalunits as nu
nu.reset_units('SI')

def diagnostics(df):
    pprint(df.columns)

def format_conversion(df, format_cl, col_ids):
    """select column data needs to be converted based on given types"""
    df_c = df[df[col_ids["format"]].isin(format_cl)]
    df_c = df_c[df_c[col_ids["param"]].notnull()]
    df_c[col_ids["param"] + ".modified"]= df_c.apply(lambda d: round( col_ids["func"] [d[col_ids["format"]]]
                                                    (float(d[col_ids["param"]])) ), axis=1)

    df[col_ids["param"] + ".modified"] = np.nan
    df.update(df_c)
    return df
