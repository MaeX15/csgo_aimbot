import pynput
def lock(aims,mouse, x, y, flag, dx_last):
    dist_list = []
    aim = 0
    mouse_pos_x, mouse_pos_y = mouse.position
    for i, det in enumerate(aims):
        c, x_center, y_center, w_w, h_h = det
        if float(c) == 0:
            continue
        dist = (x * float(x_center) - mouse_pos_x) ** 2 + (y * float(y_center) - mouse_pos_y) ** 2
        # dist = (x * float(w_w))*(y * float(h_h)) ** 2
        if len(dist_list) == 0 or dist < dist_list[-1]:
            dist_list.append(dist)
            aim = i
        # if float(c) == 1.:
        #     aim = i
        #     break
    det_aim = aims[aim]
    c, x_c, y_c, width, height = det_aim
    x_cen = x * float(x_c)
    y_cen = y * float(y_c)
    dx = x_cen - mouse_pos_x
    if dx_last != 0 and dx/dx_last > 1:
        mouse.position = (x_cen + 0.5 * dx, y_cen)
    else:
        mouse.position = (x_cen, y_cen)
    dx_last = dx
    if flag % 10 == 0:
        mouse.click(pynput.mouse.Button.left)
        dx_last = 0
    return dx_last