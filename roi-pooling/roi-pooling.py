import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    pooled_rois = []

    for x1, y1, x2, y2 in rois:
        roi_h = y2 - y1
        roi_w = x2 - x1
        pooled = []

        for i in range(output_size):
            row = []
            for j in range(output_size):
                h_start = y1 + math.floor(i * roi_h / output_size)
                h_end = y1 + math.floor((i + 1) * roi_h / output_size)
                w_start = x1 + math.floor(j * roi_w / output_size)
                w_end = x1 + math.floor((j + 1) * roi_w / output_size)

                if h_end == h_start:
                    h_end = h_start + 1
                if w_end == w_start:
                    w_end = w_start + 1

                max_val = float('-inf')
                for y in range(h_start, h_end):
                    for x in range(w_start, w_end):
                        max_val = max(max_val, feature_map[y][x])

                row.append(max_val)
            pooled.append(row)

        pooled_rois.append(pooled)

    return pooled_rois