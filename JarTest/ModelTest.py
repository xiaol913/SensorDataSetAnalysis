from sklearn.externals import joblib


def test_pkl():
    clf = joblib.load('feature.pkl')
    test = [[4.7234607, 5.2234, -5.903802],
            [-2.638092, 5.004593, 7.5350037],
            [3.711, -3.117, -1.65]]
    predictions = clf.predict(test)
    print(predictions)


test_pkl()
