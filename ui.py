import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import leastsq
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline
import pandas as pd
import svgwrite

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

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

    svg_path = 'output.svg'
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
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

    return svg_path

def save_to_csv(paths_XYs):
    all_coords = []
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            for point in XY:
                # Ensure there are 4 columns
                if len(point) < 4:
                    point = np.pad(point, (0, 4 - len(point)), constant_values=np.nan)
                all_coords.append(point)
    df = pd.DataFrame(all_coords, columns=['x', 'y', 'z', 'w'])
    csv_path = 'coordinates.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

def is_line(XY):
    if len(XY) < 2:
        return False
    distances = cdist(XY, XY)
    return np.allclose(np.max(distances) - np.min(distances), 0, atol=1e-2)

def fit_line(XY):
    model = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor())
    model.fit(XY[:, 0].reshape(-1, 1), XY[:, 1])
    line = model.predict(XY[:, 0].reshape(-1, 1))
    return np.column_stack([XY[:, 0], line])

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

def fit_polygon(XY):
    centroid = np.mean(XY, axis=0)
    angles = np.arctan2(XY[:, 1] - centroid[1], XY[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return XY[sorted_indices]

def regularize_shapes(path_XYs):
    regularized_paths = []
    for XYs in path_XYs:
        regularized_shape = []
        for XY in XYs:
            if is_line(XY):
                regularized_shape.append(fit_line(XY))
            elif len(XY) >= 5:
                regularized_shape.append(fit_circle_ellipse(XY))
            else:
                regularized_shape.append(fit_polygon(XY))
        regularized_paths.append(regularized_shape)
    return regularized_paths

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Polyline Plotter and SVG Generator"), className="text-center my-4")),
    
    dbc.Row(dbc.Col(dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ))),

    dbc.Row(dbc.Col(html.Div(id='output-plot', className="my-4 text-center"))),
    
    dbc.Row([
        dbc.Col(dbc.Button("Download SVG", id="download-svg-btn", color="primary", className="me-2", n_clicks=0)),
        dbc.Col(dbc.Button("Download PNG", id="download-csv-btn", color="secondary", className="me-2", n_clicks=0))
    ], className="text-center my-4"),
    
    dcc.Download(id="download-svg"),
    dcc.Download(id="download-csv")
])

@app.callback(
    Output('output-plot', 'children'),
    Input('upload-data', 'contents')
)
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        np_array = np.genfromtxt(io.StringIO(decoded.decode('utf-8')), delimiter=',')
        paths_XYs = read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Regularize shapes before plotting
        regularized_paths = regularize_shapes(paths_XYs)
        
        img_b64 = plot(regularized_paths)
        return html.Img(src=f'data:image/png;base64,{img_b64}', style={"max-width": "100%", "height": "auto"})
    return "No file uploaded yet."

@app.callback(
    Output("download-svg", "data"),
    Output("download-csv", "data"),
    Input("download-svg-btn", "n_clicks"),
    Input("download-csv-btn", "n_clicks"),
    State('upload-data', 'contents')
)
def download_files(n_svg, n_csv, contents):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        paths_XYs = read_csv(io.StringIO(decoded.decode('utf-8')))
        
        if n_svg > 0:
            svg_path = polylines2svg(paths_XYs)
            return dcc.send_file(svg_path), None
        
        if n_csv > 0:
            csv_path = save_to_csv(paths_XYs)
            return None, dcc.send_file(csv_path)

    return None, None

if __name__ == "__main__":
    app.run_server(debug=False)
