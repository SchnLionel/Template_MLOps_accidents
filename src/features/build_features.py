def add_new_columns(df, nb_victim, nb_vehicules):
    """
    Merges victim and vehicle counts into the main dataframe.
    """
    df = df.merge(nb_victim, on="Num_Acc", how="inner")
    df.rename({"count": "nb_victim"}, axis=1, inplace=True)
    df = df.merge(nb_vehicules, on="Num_Acc", how="inner")
    df.rename({"count": "nb_vehicules"}, axis=1, inplace=True)
    return df


def modif_target_variable(df):
    """
    Modifies the target variable: 1 = prioritary, 0 = non-prioritary
    """
    df['grav'].replace([2, 3, 4], [0, 1, 1], inplace=True)
    return df


def replace_values(df):
    """
    Replaces -1 and 0 values with NaN in specific columns.
    """
    import numpy as np
    col_to_replace0_na = ["trajet", "catv", "motor"]
    col_to_replace1_na = ["trajet", "secu1", "catv", "obsm", "motor", "circ", "surf", "situ", "vma", "atm", "col"]
    df[col_to_replace1_na] = df[col_to_replace1_na].replace(-1, np.nan)
    df[col_to_replace0_na] = df[col_to_replace0_na].replace(0, np.nan)
    return df


def drop_columns(df):
    """
    Drops unnecessary columns from the dataframe.
    """
    list_to_drop = ['senc', 'larrout', 'actp', 'manv', 'choc', 'nbv', 'prof', 'plan', 'Num_Acc', 
                    'id_vehicule', 'num_veh', 'pr', 'pr1', 'voie', 'trajet', "secu2", "secu3", 
                    'adr', 'v1', 'lartpc', 'occutc', 'v2', 'vosp', 'locp', 'etatp', 'infra', 'obs']
    df.drop(list_to_drop, axis=1, inplace=True)
    return df


def drop_nans(df):
    """
    Drops rows with NaN values in critical columns.
    """
    col_to_drop_lines = ['catv', 'vma', 'secu1', 'obsm', 'atm']
    df = df.dropna(subset=col_to_drop_lines, axis=0)
    return df
