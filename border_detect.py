from shapely.geometry import LineString, Polygon
from shapely.ops import split

def get_area_below_line(polygon_points, line):
    """
    Calculates the area of the part of the polygon that is below the given line.

    Args:
    polygon_points (list of tuples): List of (x, y) tuples representing the vertices of the polygon.
    line (shapely LineString): line that represents the border between V1 and V2

    Returns:
    float: Area of the part of the polygon below the line.
    """
    # Create a Polygon object from the list of points
    polygon = Polygon(polygon_points)

    if polygon.crosses(line):
        # Split the polygon by the line
        split_polygons = split(polygon, line) 
        area_below_line = 0.0
        # Determine which part is below the line and calculate its area
        for poly in split_polygons:
            # Check if the centroid of the polygon is below the line
            if poly.centroid.y < min(line_start[1], line_end[1]):
                area_below_line += poly.area
        return area_below_line
    else:
        return None

# Example usage
polygon_points = [(0, 0), (4, 0), (4, 4), (0, 4)]  # Square polygon
line_start = (2, -1)
line_end = (2, 5)
border_line = LineString([line_start, line_end])

area_below = get_area_below_line(polygon_points, border_line)
print("Area of the part of the polygon below the line:", area_below)
