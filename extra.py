# fill missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(fill_value=np.nan, startegy='mean')
X = imputer.fit_transform(df)
# Because it returns a numpy array, to read it, we can convert it back to the data frame.

# drop missing values
dropedDf = df.dropna()
