# Define the train and test arrays
train = [train1, train2, train3]
test = [test1, test2, test3]

# Set the embedding parameters
embedding_dim_range = range(3, 5)
tau_range = np.arange(1, 1.1, 0.0005)

# Initialize variables
min_RAE = np.inf
min_embedding_dim = None
min_tau = None
min_train = None
min_test = None


# Loop over the train and test arrays
for i in range(len(train)):
    ts_train = pd.read_csv(train[i], index_col='date', parse_dates=True)
    ts_train = ts_train.resample('D').mean()

    ts_test = pd.read_csv(test[i], index_col='date', parse_dates=True)

    ts_combined = pd.concat([ts_train, ts_test])

    # Loop over the embedding parameters
    for embedding_dim in embedding_dim_range:     
        for tau in tau_range:
            embedding = np.column_stack([ts_combined.values[i:(len(ts_combined)-embedding_dim+1+i)] for i in range(embedding_dim)])
            y0 = embedding[-1]

            n_pred = len(ts_test)

            def f(y, t):
                idx = int(t/tau) - embedding_dim + 1
                if idx >= len(embedding):
                    return np.zeros(embedding_dim)
                else:
                    #A = 25 # Amplitude
                    f = 1/365 # Frequency (one cycle per year)
                    phi = 365 # Phase shift
                    return  A*np.sin(2*np.pi*f*t + phi) +embedding[idx+1] - embedding[idx]
            


            y_pred = odeint(f, y0, np.arange(1, n_pred+1))

            index = pd.date_range(start=ts_test.index[0], periods=len(y_pred), freq='D')
            forecasted_ts = pd.DataFrame(data=y_pred[:, 0], index=index, columns=['meantemp'])

            RAE = np.mean(np.abs(y_pred[:, 0] - ts_test['meantemp']) / ts_test['meantemp']) * 100
            print(f"Train: {[i + 1]}, Test: {[i + 1]}, Embedding dim: {embedding_dim}, Tau: {tau:.4f}, RAE: {RAE:.2f}%")

            if RAE < min_RAE:
                min_RAE = RAE
                min_embedding_dim = embedding_dim
                min_tau = tau
                min_train = train[i]
                min_test = test[i]
                print(f"\nNew minimum RAE: {RAE:.2f}")

print(f"\nMinimum RAE: {min_RAE:.2f}")
print(f"Best embedding dim: {min_embedding_dim}")
print(f"Best tau: {min_tau:.4f}")
print(f"Train file: {min_train}")
print(f"Test file: {min_test}")
