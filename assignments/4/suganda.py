# Arif Suganda
# F5522530002
# Import required libraries
import warnings, requests, zipfile, io  
# warnings for suppressing warnings
# requests for downloading data
# zipfile for extracting zip files
# io for handling in-memory byte streams

# Suppress warning messages for cleaner output
warnings.simplefilter('ignore')

# Import required libraries
import pandas as pd
from scipy.io import arff
# pandas for data manipulation
# scipy.arff for loading ARFF files
# ARRF stands for Attribute-Relation File Format, a text-based file format used to describe datasets 
# Details on https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/

# Define the URL to the dataset (vertebral column dataset from UCI repository)
# Details on https://archive.ics.uci.edu/dataset/212/vertebral+column
f_zip = 'https://archive.ics.uci.edu/static/public/212/vertebral+column.zip'

# Download the dataset using requests (stream=True allows downloading in chunks)
r = requests.get(f_zip, stream=True)

# Open the zip file from the downloaded content (stored in memory)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))

# Extract all contents of the zip file to the current working directory
Vertebral_zip.extractall()

# Load the ARFF dataset (column_2C_weka.arff file extracted earlier)
data = arff.loadarff('column_2C_weka.arff')

# Convert the ARFF data into a pandas DataFrame for analysis
df = pd.DataFrame(data[0])

# Display the shape of the dataset (number of rows and columns)
print(df.shape)

# Display the column names in the dataset
print(df.columns)

# Display data types of each column
print(df.dtypes)

# Display summary statistics (count, average, mean, distibution min, max, etc.) for one column
print(df['pelvic_incidence'].describe())

# Display summary statistics (count, average, mean, distibution min, max, etc.) for all columns
print(df.describe())

# Import matplotlib for plotting (term for graph)
import matplotlib.pyplot as plt

# Command for Jupyter notebook to render matplotlib plots directly inside the notebook
# %matplotlib inline

# Plot all numeric columns as line graphs
# Purpose: Shows the distribution of values for each class to see trends within the data
df.plot()
# Display the plot
plt.show()

# Plot density plots (KDE) for each numeric column, arranged in a 4x2 grid
# Purpose: Shows the shape of the distribution of all classes
# Details on https://www.data-to-viz.com/graph/density.html
df.plot(kind='density', subplots=True, layout=(4,2), figsize=(12,12), sharex=False)
# Display the plot
plt.show()

# Plot density curve for 'degree_spondylolisthesis' 
# Purpose: Shows the distribution of one class
# degree_spondylolisthesis is a class within the dataset
df['degree_spondylolisthesis'].plot.density()
# Display the plot
plt.show()

# Plot histogram for 'degree_spondylolisthesis'
# Purpose: Shows how often values fall into certain ranges from low to high
# Details on https://en.wikipedia.org/wiki/Histogram
df['degree_spondylolisthesis'].plot.hist()
# Display the plot
plt.show()

# Plot boxplot for 'degree_spondylolisthesis'
# Purpose: Shows the median, spread, and possible outliers of one class
# Details on https://en.wikipedia.org/wiki/Box_plot
df['degree_spondylolisthesis'].plot.box()
# Display the plot
plt.show()

# Count the number of samples per class (Normal, Abnormal)
# Purpose: Check for class imbalance in the dataset
print(df['class'].value_counts())

# Map class labels from bytes to integers (Abnormal to 1, Normal to 0)
class_mapper = {b'Abnormal': 1, b'Normal': 0}
df['class'] = df['class'].replace(class_mapper)

# Scatter plot: class (x-axis) vs degree_spondylolisthesis (y-axis)
# Purpose: Shows how well the feature separates between classes
# Details on https://en.wikipedia.org/wiki/Scatter_plot
df.plot.scatter(y='degree_spondylolisthesis', x='class')
# Display the plot
plt.show()

# Group data by class and create boxplots for each group
# Purpose: Shows the median, spread, and possible outliers of all classess
df.groupby('class').boxplot(fontsize=20, rot=90, figsize=(20,10), patch_artist=True)
# Display the plot
plt.show()

# Compute correlation matrix of all numeric columns
# Details on https://en.wikipedia.org/wiki/Correlation
corr_matrix = df.corr()

# Show correlations with the 'class' column, sorted by strength
# Purpose: Identify features that are most correlated with the target classess
print(corr_matrix["class"].sort_values(ascending=False))

# Create scatter matrix of all numeric columns
# Purpose: Shows relationships between different measurements to indentify trends within the data
pd.plotting.scatter_matrix(df, figsize=(12,12))
# Display the plot
plt.show()

# Import seaborn for advanced visualization
import seaborn as sns

# Create a figure with custom size
fig, ax = plt.subplots(figsize=(10, 10))

# Define a custom color palette (Brown-Blue-Green with 10 levels)
colormap = sns.color_palette("BrBG", 10)

# Create a heatmap of the correlation matrix
# Purpose: Shows an overview of relationships between all numeric features (darker = stronger relationship)
# Details on https://en.wikipedia.org/wiki/Heat_map
sns.heatmap(corr_matrix, cmap=colormap, annot=True, fmt=".2f")

# Display the heatmap
plt.show()
