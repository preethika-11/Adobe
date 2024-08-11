import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import dash_table
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import leastsq
import svgwrite
import cairosvg
import pandas as pd
from scipy.spatial.distance import cdist
from dash.dcc import  send_file
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

# Function to detect if the shape is a line
def is_line(XY, threshold=0.01):
    if len(XY) < 2:
        return False
    distances = np.diff(XY, axis=0)
    angles = np.arctan2(distances[:, 1], distances[:, 0])
    return np.std(angles) < threshold

# Function to fit a line using linear regression
def fit_line(XY):
    model = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor())
    model.fit(XY[:, 0].reshape(-1, 1), XY[:, 1])
    line = model.predict(XY[:, 0].reshape(-1, 1))
    return np.column_stack([XY[:, 0], line])

# Function to check if the shape is a circle
def check_if_circle(XY, tolerance=0.1):
    center = np.mean(XY, axis=0)
    distances = np.sqrt((XY[:, 0] - center[0])**2 + (XY[:, 1] - center[1])**2)
    mean_radius = np.mean(distances)
    return np.max(np.abs(distances - mean_radius)) < tolerance * mean_radius

# Function to fit circles and ellipses
def fit_circle_ellipse(XY):
    def calc_r(xc, yc):
        """ Calculate the distance of each data point from the center (xc, yc) """
        return np.sqrt((XY[:, 0] - xc) ** 2 + (XY[:, 1] - yc) ** 2)

    def f_2(c):
        """ Calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_r(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(XY[:, 0]), np.mean(XY[:, 1])
    center, _ = leastsq(f_2, center_estimate)
    radius = calc_r(*center).mean()

    theta = np.linspace(0, 2 * np.pi, 100)
    x_fit = center[0] + radius * np.cos(theta)
    y_fit = center[1] + radius * np.sin(theta)

    return np.column_stack([x_fit, y_fit])

# Function to detect and fit polygons
def fit_polygon(XY):
    centroid = np.mean(XY, axis=0)
    angles = np.arctan2(XY[:, 1] - centroid[1], XY[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return XY[sorted_indices]

# Function to reorient shapes from bottom-to-top to top-to-bottom
def reorient_shapes(path_XYs):
    max_y = max([max(XY[:, 1]) for XYs in path_XYs for XY in XYs])
    min_y = min([min(XY[:, 1]) for XYs in path_XYs for XY in XYs])
    if max_y < 0:  # Shapes are above the x-axis, implying top-to-bottom order
        return path_XYs
    else:
        flipped_paths = []
        for XYs in path_XYs:
            flipped_XYs = []
            for XY in XYs:
                flipped_XY = np.copy(XY)
                flipped_XY[:, 1] = max_y - (flipped_XY[:, 1] - min_y)
                flipped_XYs.append(flipped_XY)
            flipped_paths.append(flipped_XYs)
        return flipped_paths

# Regularization: Identify and approximate regular shapes with conditions
def regularize_shapes(path_XYs):
    regularized_paths = []
    for XYs in path_XYs:
        regularized_shape = []
        for XY in XYs:
            if is_line(XY, threshold=0.01):  # Detect and fit lines with a stricter threshold
                fitted_line = fit_line(XY)
                if np.linalg.norm(fitted_line[-1] - fitted_line[0]) > 10:  # Keep only lines longer than 10 units
                    regularized_shape.append(fitted_line)
            elif check_if_circle(XY):  # Detect and fit circles
                fitted_circle = fit_circle_ellipse(XY)
                radius = np.mean(np.sqrt((fitted_circle[:, 0] - np.mean(fitted_circle[:, 0]))**2 +
                                         (fitted_circle[:, 1] - np.mean(fitted_circle[:, 1]))**2))
                if radius > 5:  # Keep only circles with a radius greater than 5 units
                    regularized_shape.append(fitted_circle)
            else:  # For other shapes, use polygon fitting or other methods
                fitted_polygon = fit_polygon(XY)
                if len(fitted_polygon) > 3:  # Keep only polygons with more than 3 vertices
                    regularized_shape.append(fitted_polygon)
                    
        if regularized_shape:  # Only append if there are any shapes left after filtering
            regularized_paths.append(regularized_shape)
    return regularized_paths

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colors[i % len(colors)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def polylines2svg(paths_XYs):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    dwg = svgwrite.Drawing('output.svg', profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colors[i % len(colors)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
    dwg.add(group)
    dwg.save()

    cairosvg.svg2png(url='output.svg', write_to='output.png', parent_width=W, parent_height=H,
                     output_width=W, output_height=H, background_color='white')

    return 'output.svg', 'output.png'

def classify_shape_category(paths_XYs):
    categories = []
    for XYs in paths_XYs:
        if len(XYs) == 1:
            categories.append("Isolated")
        elif len(XYs) > 1 and any(np.allclose(XYs[0][0], XY[-1]) for XY in XYs):
            categories.append("Fragmented")
        elif identify_symmetry(XYs[0]):
            categories.append("Connected Occlusion")
        else:
            categories.append("Disconnected Occlusion")
    return categories

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("CURVETOPIA: A Journey into the World of Curves"), className="text-center my-4")),
    dbc.Row(dbc.Col(dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=False,
    ))),
    dbc.Row(dbc.Col(html.Div(id='output-data-upload'))),
    dbc.Row(dbc.Col(html.Img(id='image'))),
    dbc.Row(dbc.Col(dcc.Download(id="download-svg"))),
    dbc.Row(dbc.Col(dcc.Download(id="download-png"))),
])


@app.callback(
    [Output('output-data-upload', 'children'),
     Output('image', 'src'),
     Output('download-svg', 'data'),
     Output('download-png', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def update_output(content, filename, date):
    if content is None:
        return '', None, None, None
    
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    path_XYs = read_csv(io.StringIO(decoded.decode('utf-8')))
    original_plot = plot(path_XYs)

    path_XYs = reorient_shapes(path_XYs)
    path_XYs = regularize_shapes(path_XYs)

    regularized_plot = plot(path_XYs)
    svg_file, png_file = polylines2svg(path_XYs)

    # Read and encode files
    with open(svg_file, 'rb') as f:
        svg_encoded = base64.b64encode(f.read()).decode('utf-8')

    with open(png_file, 'rb') as f:
        png_encoded = base64.b64encode(f.read()).decode('utf-8')

    # Prepare download data
    svg_data = {
        "content": f"data:image/svg+xml;base64,{svg_encoded}",
        "filename": "output.svg"
    }

    png_data = {
        "content": f"data:image/png;base64,{png_encoded}",
        "filename": "output.png"
    }

    return (
        html.Div([
            html.H5(f"Original Plot from {filename}"),
            html.Img(src='data:image/png;base64,' + original_plot),
            html.H5(f"Regularized Plot from {filename}"),
            html.Img(src='data:image/png;base64,' + regularized_plot),
        ]),
        'data:image/png;base64,' + regularized_plot,
        svg_data,
        png_data
    )


if __name__ == '__main__':
    app.run_server(debug=True)
