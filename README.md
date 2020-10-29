# kalman_filter
Kalman Filter that i used to take coordinates from DeepLabCut (DLC) to smooth them out.

Creates a separate kalman filter for each body part to keep track of the animal.

Usage:
call create_kalman_filters with n amount of 2D points (four in this example). The zeros represtent the initial points and should be picked accordingly.

```python
kalman_filters = create_kalman_filters(np.array([[0,0,0,0],
                                                 [0,0,0,0]]))
```
For each time frame call the function predict_and_update with n (x,y) points (four in this example) and the kalman_filters that are returned by create_kalman_filters.
```python
filtered_coords = predict_and_update(np.array([[snout_x,l_ear_x,r_ear_x,tail_x],
                                               [snout_y,l_ear_y,r_ear_y,tail_y]]),
                                               kalman_filters)
```
Change the time constant T if you want to change how fast the Kalman filter is.

