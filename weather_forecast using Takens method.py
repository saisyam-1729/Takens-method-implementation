# Load the training dataset and resample to daily frequency
ts_train = pd.read_csv('', index_col='date', parse_dates=True)
ts_train = ts_train.resample('D').mean()

# Load the test dataset
ts_test = pd.read_csv('', index_col='date', parse_dates=True)

# Combine the training and test datasets
ts_combined = pd.concat([ts_train, ts_test])

# Set the embedding parameters
embedding_dim = 
tau = 

# Create the embedding matrix
embedding = np.column_stack([ts_combined.values[i:(len(ts_combined)-embedding_dim+1+i)] for i in range(embedding_dim)])

# Extract the initial condition for prediction
y0 = embedding[-1]

# Set the number of days to predict
n_pred = len(ts_test)+730

# Define the function for the ODE system
def f(y, t):
    idx = int(t/tau) - embedding_dim + 1
    if idx >= len(embedding):
        return np.zeros(embedding_dim)
    else:
        A = 25 # Amplitude
        f = 1/365 # Frequency (one cycle per year)
        phi = 0 # Phase shift
        return  A*np.sin(2*np.pi*f*t + phi) +embedding[idx+1] - embedding[idx]

def f(y, t):
    idx = int(t/tau) - embedding_dim + 1
    if idx >= len(embedding):
        return np.zeros(embedding_dim)
    else:
        return embedding[idx+1] - embedding[idx]

# Integrate the ODE to predict the values
y_pred = odeint(f, y0, np.arange(1, n_pred+1))

# Create a DataFrame of the predicted values
index = pd.date_range(start=ts_test.index[0], periods=len(y_pred), freq='D')
forecasted_ts = pd.DataFrame(data=y_pred[:, 0], index=index, columns=['meantemp'])

# Calculate the MAE between the predicted and test values
MAE = np.mean(np.abs(forecasted_ts['meantemp'] - ts_test['meantemp']))
print(f"MAE: {MAE:.2f}")

RAE = np.mean(np.abs(y_pred[:, 0] - ts_test['meantemp']) / ts_test['meantemp']) * 100
print(f" RAE: {RAE:.2f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_train.index, y=ts_train['meantemp'], name='Training Data'))
fig.add_trace(go.Scatter(x=forecasted_ts.index, y=forecasted_ts['meantemp'], name='Predicted Values'))
#fig.add_trace(go.Scatter(x=ts_test.index, y=ts_test['meantemp'], name='Test Data'))
fig.update_layout(title=f'Mean Temperature Prediction\nMAE: <span style="color:red">{MAE:.2f}</span>, RAE: <span style="color:blue">{RAE:.2f}%</span>', xaxis_title='Date', yaxis_title='Puttaparthi Temperature (Celsius)')
fig.show()
