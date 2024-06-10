from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def train(data_frame):
    categorical = ['PULocationID', 'DOLocationID']
    train_dict = data_frame[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dict)
    y_train = data_frame['duration'].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    rmse = mean_squared_error(y_train, y_pred, squared=False)
    print(f'RMSE: {rmse}')

    return dv, lr, lr.intercept_