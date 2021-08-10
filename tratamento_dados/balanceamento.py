from imblearn.under_sampling import ClusterCentroids
import time
# Realiza o balanceamento dos dados
def balanceamento(X, y):
    t = time.time()    
    print('O balancemaneo dos dados iniciou...')
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X, y)
    
    segundos = time.time() - t
    print('O balanceamento dos dados terminou e levou ', segundos, ' segundos!')

    return [X_resampled, y_resampled]