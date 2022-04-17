def intersect1d(x,y,x_target):
    f = interp1d(x,y)
    return (x_target, f(x_target))

def pinch_lr(x_data, y_data, range_lr = (0,10)):
    '''
    Find the pinch-off from Linear Regression (LR) around an inflexion point
    infl: Final point of linear regression interpolation
    range_lr: Nulber of points to do linear regression interpolation
    '''
    x_data_lr = x_data[range_lr[0]:range_lr[1]].copy()
    y_data_lr = y_data[range_lr[0]:range_lr[1]].reshape((-1, 1)).copy()
    # Do linear regression around inflexion point
    model = LinearRegression(copy_X=False)
    model.fit(y_data_lr, x_data_lr)
    y_data_lr = np.append([0], y_data[range_lr[0]:range_lr[1]].copy())
    y_data_lr = y_data_lr.reshape((-1, 1))
    x_data_lr = model.predict(y_data_lr)
    return x_data_lr, y_data_lr, x_data_lr[0]