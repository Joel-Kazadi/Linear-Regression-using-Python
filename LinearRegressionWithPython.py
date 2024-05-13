# 0. IMPORTING LIBRAIRIES

#pip install bokeh
import os
import unittest
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LinearRegression
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Float, String, text
from sqlalchemy import create_engine, Table, MetaData


# 1. DEFINING THE DATABASE SCHEMA

def create_database(database_path):
    """
    Create a SQLite database and tables to store training data,
    test data results, and ideal functions.
    """
    engine = create_engine(f'sqlite:///{database_path}')
    metadata = MetaData()
    
    # Define the training data table
    train_table = Table('train_data', metadata,
                        Column('x', Float, primary_key=True),
                        Column('y1', Float),
                        Column('y2', Float),
                        Column('y3', Float),
                        Column('y4', Float))
    
    # Define the ideal functions table
    ideal_table = Table('ideal_functions', metadata,
                        Column('function_name', String, primary_key=True),
                        Column('x', Float),
                        Column('y', Float))

    # Define the test results table
    test_results_table = Table('test_results', metadata,
                                Column('x', Float),
                                Column('y', Float),
                                Column('function_name', String),
                                Column('deviation', Float))
    
    # Create the tables in the SQLite database
    metadata.create_all(engine)
    
    # Return engine and session
    Session = sessionmaker(bind=engine)
    session = Session()
    return engine, session


# 2. CODE IMPLEMENTATION

class DataHandler:
    """
    Base class for data handling.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """
        Load data from a CSV file. 
        To be implemented in derived classes.
        """
        raise NotImplementedError("Implement this method in derived classes")
    
    def preprocess_data(self):
        """
        Preprocess data.
        """
        raise NotImplementedError("Implement this method in derived classes")


class TrainDataHandler(DataHandler):
    """
    Class for handling training data.
    """
    def load_data(self):
        """
        Load training data from a CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
        except Exception as e:
            raise Exception(f"Failed to load training data: {str(e)}")
    
    def preprocess_data(self):
        """
        Preprocess the training data.
        """
        self.data['x'] = self.data['x'].values.reshape(-1, 1)
        return self.data


class TestDataHandler(DataHandler):
    """
    Class for handling test data.
    """
    def load_data(self):
        """
        Load test data from a CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
        except Exception as e:
            raise Exception(f"Failed to load test data: {str(e)}")
    
    def preprocess_data(self):
        """
        Preprocess the test data.
        """
        self.data['x'] = self.data['x'].values.reshape(-1, 1)
        return self.data


def linear_regression(x, y):
    """
    Perform linear regression and
    return the model and the sum of squared deviations.
    """
    
    # Reshape x if it is a 1D array
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    deviations_squared = (y - y_pred)**2
    sum_deviations_squared = np.sum(deviations_squared)
    return model, sum_deviations_squared


class IdealFunctionSelector:
    """
    Class for selecting ideal functions.
    """
    def __init__(self, train_df):
        self.train_df = train_df
    
    def select_ideal_functions(self):
        """
        Select the four ideal functions
        based on sum of squared deviations.
        """
        x = self.train_df['x'].values
        ideal_functions = []
        sum_deviations_squared_list = []
        
        # Perform linear regression for each of the 50 y columns
        for i in np.arange(1, self.train_df.shape[1]):
            y = self.train_df[f'y{i}'].values
            model, sum_deviations_squared = linear_regression(x, y)
            ideal_functions.append((model, sum_deviations_squared, f'y{i}'))
            sum_deviations_squared_list.append(sum_deviations_squared)
        # Sort the ideal functions by their sum of squared deviations
        ideal_functions.sort(key=lambda x: x[1])
        
        # Select the top-5 functions
        return ideal_functions[:5]



class TestMapping:
    """
    Class for mapping test data to ideal functions.
    """
    def __init__(self, test_df, ideal_functions):
        self.test_df = test_df
        self.ideal_functions = ideal_functions
    
    def map_test_data(self):
        """
        Map the test data to the four chosen ideal functions
        (without relaxation).
        """
        x_test = self.test_df['x'].values
        y_test = self.test_df['y'].values
        mappings, deviations = [], []
        
        # Iterate over each test data point
        for x_val, y_val in zip(x_test, y_test):
            min_deviation = float('inf')
            best_function = None
            
            # Iterate over each ideal function
            for model, _, function_name in self.ideal_functions[:4]:
                # Predict y-value for the given test x-value
                y_pred = model.predict([[x_val]])
                # Calculate the absolute deviation
                deviation = abs(y_val - y_pred)
                
                # Check if this deviation is the smallest found so far
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_function = function_name
            
            # Check if the best deviation meets the threshold
            if min_deviation < float('inf'):
                # Allowable deviation: standard deviation from function
                allowable_deviation = np.sqrt(min_deviation)
                
                # Accept mapping if deviation is within range
                if min_deviation <= allowable_deviation:
                    mappings.append(best_function)
                    deviations.append(min_deviation)
                else:
                    mappings.append('Unassigned')
                    deviations.append(min_deviation)
            else:
                # If no suitable function found, mark it as 'Unassigned'
                mappings.append('Unassigned')
                deviations.append(min_deviation)
            
        return mappings, deviations

    def relaxed_map_test_data(self):
        """
        Map the test data to the four chosen ideal functions
        (with relaxation).
        """
        x_test = self.test_df['x'].values
        y_test = self.test_df['y'].values
        mappings, deviations = [], []
        
        # Iterate over each test data point
        for x_val, y_val in zip(x_test, y_test):
            min_deviation = float('inf')
            best_function = None
            
            # Iterate over each ideal function
            for model, _, function_name in self.ideal_functions[:4]:
                # Predict y-value for the test x-value
                y_pred = model.predict([[x_val]])
                # Calculate absolute deviation
                deviation = abs(y_val - y_pred)
                
                # Check if this deviation is the smallest found
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_function = function_name
            
            # Allow test point to be assigned based on relaxed criteria
            if best_function:
                # If minimum deviation meets adjusted threshold
                mappings.append(best_function)
                deviations.append(min_deviation)
            else:
                # If none match, assign smallest deviation
                mappings.append('Unassigned')
                deviations.append(min_deviation)
        
        return mappings, deviations


def save_to_database(session, train_df, ideal_functions, test_results):
    """
    Save training data, ideal functions, and test results
    to the SQLite database.
    """
    # Insert training data into the train_data table
    for index, row in train_df.iterrows():
        statement = text('INSERT INTO train_data (:x, :y1, :y2, :y3, :y4)')
        session.execute(statement, {'x': row['x'],
                                    'y1': row['y1'],
                                    'y2': row['y2'],
                                    'y3': row['y3'],
                                    'y4': row['y4']})

    # Insert ideal functions into the ideal_functions table
    for function_name, model, _ in ideal_functions:
        x_values = train_df['x'].values
        y_predicted = model.predict(x_values.reshape(-1, 1))
        for x, y in zip(x_values, y_predicted):
            statement = text('INSERT INTO ideal_funct (:funct_name, :x, :y)')
            session.execute(statement,
                            {'function_name': function_name, 'x': x, 'y': y}
                            )
    
    # Insert test results into the test_results table
    for result in test_results:
        x_val, y_val, function_name, deviation = result
        statement = text('INSERT INTO test_results (:x,:y,:funct_name,:dev)')
        session.execute(statement, {'x_val': x_val,
                                    'y_val': y_val,
                                    'function_name': function_name,
                                    'deviation': deviation})
    
    # Commit the transaction
    session.commit()


def visualize_results(train_df, ideal_functions, test_df, mappings):
    """
    Visualize the selected ideal functions and
    the test data mapping using Bokeh.
    """
    
    # Create the output file for Bokeh visualization
    output_file("visualization.html")

    # Create a new Bokeh figure
    p = figure(title="Regression Mapping: Ideal Functions vs Test Data",
               x_axis_label='X', y_axis_label='Y')

    # Define color mapping for each function and unassigned points
    ideal_function_names = [func[2] for func in ideal_functions[:4]]
    color_mapping = {
        ideal_function_names[0]: 'maroon',
        ideal_function_names[1]: 'magenta',
        ideal_function_names[2]: 'lawngreen',
        ideal_function_names[3]: 'cyan',
        'Unassigned': 'gray'
    }

    # Plot each ideal function with colors consistent with the color mapping
    x = train_df['x'].values
    for model, _, function_name in ideal_functions[:4]:
        
        # Predict y-values using the model for each x-value
        y_pred = model.predict(x.reshape(-1, 1))
        
        # Get the color for the function from the color mapping dictionary
        color = color_mapping[function_name]
        
        # Plot the ideal function line with the specified color
        p.line(x,y_pred,color=color,legend_label=function_name,line_width=2)

    # Add color mapping based on the mappings parameter
    test_df['color']=[color_mapping.get(mapping,'gray')for mapping in mappings]
    
    # Plot test data points with the corresponding colors and legend field
    p.circle(test_df['x'], test_df['y'],
             size=10, color=test_df['color'], legend_field='color')
    
    # Increase font size of axis labels, tick labels, and title
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.title.text_font_size = "16pt"
    
    # Show the plot
    show(p)


# 3. UNIT TEST

class TestLinearRegressionFunctions(unittest.TestCase):
    
    def setUp(self):
        # Set up the data for testing
        self.train_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y1': [1, 2, 3, 4, 5],
            'y2': [2, 4, 6, 8, 10],
            'y3': [1, 4, 9, 16, 25],
            'y4': [1, 1, 1, 1, 1]
        })
        
        self.test_data = pd.DataFrame({
            'x': [2, 4, 6],
            'y': [2, 4, 6]
        })

        # Create instances of data handlers
        self.train_data_handler = TrainDataHandler('path/to/train_data.csv')
        self.test_data_handler = TestDataHandler('path/to/test_data.csv')
        # Use setUp to initialize data with test data
        self.train_data_handler.data = self.train_data
        self.test_data_handler.data = self.test_data

        # Create instance of IdealFunctionSelector
        self.ideal_select=IdealFunctionSelector(self.train_data_handler.data)
        
        # Select ideal functions
        self.ideal_functions = self.ideal_select.select_ideal_functions()
        
        # Create instance of TestMapping
        self.test_mapping = TestMapping(self.test_data_handler.data,
                                        self.ideal_functions)

    def test_linear_regression(self):
        # Test linear regression function
        x = self.train_data['x'].values.reshape(-1, 1)
        y = self.train_data['y1'].values
        model, sum_dev_squared = linear_regression(x, y)
        y_pred = model.predict(x)
        expected_dev_squared = np.sum((y - y_pred) ** 2)
        self.assertEqual(sum_dev_squared, expected_dev_squared)
    
    def test_select_ideal_functions(self):
        # Test ideal function selection
        ideal_functions = self.ideal_select.select_ideal_functions()
        # Should return 4 functions
        self.assertEqual(len(ideal_functions), 4)
        # The functions should be sorted by sum of squared deviations
        sum_dev_squared_list = [ideal[1] for ideal in ideal_functions]
        self.assertTrue(all(x<=y for x, y in zip(sum_dev_squared_list,
                                                 sum_dev_squared_list[1:])))

    def test_map_test_data(self):
        # Test mapping test data to ideal functions
        mappings, deviations = self.test_mapping.map_test_data()
        # Return mappings and deviations matching test data length
        self.assertEqual(len(mappings), len(self.test_data))
        self.assertEqual(len(deviations), len(self.test_data))

if __name__ == '__main__':
    unittest.main()


# 4. DISPLAYING OUTPUTS

# Create SQLite database
database_path = (os.getcwd() + '/mydatabase.db').replace('\\', '/')
engine, session = create_database(database_path)

# Instantiate the data handlers
directory_path = (os.getcwd() + '/Datasets/').replace('\\', '/')
train_path = os.path.join(directory_path, 'ideal.csv')
test_path = os.path.join(directory_path, 'test.csv')
train, test = TrainDataHandler(train_path), TrainDataHandler(test_path)

# Load and preprocess the data
train.load_data()
train_df = train.preprocess_data()
test.load_data()
test_df = test.preprocess_data()

# Select ideal functions
ideal_selector = IdealFunctionSelector(train_df)
ideal_functions = ideal_selector.select_ideal_functions()

# Map test data to ideal functions (with and without relaxation)
test_mapping = TestMapping(test_df, ideal_functions)
mappings, deviations = test_mapping.map_test_data()
rel_mappings, rel_deviations = test_mapping.relaxed_map_test_data()
    
# Visualize results
visualize_results(train_df, ideal_functions, test_df, mappings)
visualize_results(train_df, ideal_functions, test_df, rel_mappings)

# Sum of Squares of Residuals for Ideal Functions

# Extracting function names and sum-of-squares values from the data
function_names = [entry[2] for entry in ideal_functions]
sum_of_squares = [entry[1] for entry in ideal_functions]
# Creating a ColumnDataSource for Bokeh
source1 = ColumnDataSource(data=dict(function_names=function_names,
                                     sum_of_squares=sum_of_squares))

# Create a new Bokeh figure
p1 = figure(title="Sum of Squares of Residuals for Ideal Functions",
           x_range=function_names, x_axis_label="Ideal Function",
           y_axis_label="Sum of Squares of Residuals",
           plot_width=800, plot_height=400)

# Increase font size of axis labels and title
p1.xaxis.major_label_text_font_size = "12pt"
p1.yaxis.major_label_text_font_size = "12pt"
p1.xaxis.axis_label_text_font_size = "14pt"
p1.yaxis.axis_label_text_font_size = "14pt"
p1.title.text_font_size = "16pt"

# Create a bar plot using vbar
p1.vbar(x='function_names', top='sum_of_squares', width=0.5,
        source=source1, color="steelblue")

# Show the plot
show(p1)

# Distribution of Values by Ideal functions (with and without relaxation)

def plot_distribution(data_list, title):
    """
    Plots the distribution of values from the list of ideal functions.
    Parameters:
        data_list: list of values (e.g., mappings or rel_mappings).
        title: title of the plot.
    """
    
    
    # Count the occurrences of each value in the list
    count_dict = Counter(data_list)
    
    # Create a ColumnDataSource from the count dictionary
    source = ColumnDataSource(data=dict(
        labels=list(count_dict.keys()),
        counts=list(count_dict.values())
    ))
    
    # Create a new Bokeh figure
    p2 = figure(title=title,
               x_axis_label="Count",
               y_axis_label="Values",
               plot_width=800, plot_height=400,
               y_range=list(count_dict.keys()))
    # Increase font size of axis labels and title
    p2.xaxis.major_label_text_font_size = "12pt"
    p2.yaxis.major_label_text_font_size = "12pt"
    p2.xaxis.axis_label_text_font_size = "14pt"
    p2.yaxis.axis_label_text_font_size = "14pt"
    p2.title.text_font_size = "16pt"
    
    # Create a horizontal bar plot using hbar
    p2.hbar(y='labels', right='counts', height=0.5,
           source=source, color="steelblue")
    
    # Show the plot
    show(p2)

# Plot 1: Distribution of Values by Ideal functions (without relaxation)
plot_distribution(mappings, "Distribution of Values")

# Plot 2: Distribution of Values by Ideal functions (with relaxation)
plot_distribution(rel_mappings, "Distribution of Values")

# Save training data, ideal functions, and test results to the database
test_results = mappings, deviations
save_to_database(session, train_df, ideal_functions, test_results)



# 5. GIT COMMANDS (To run on Anaconda PowerShell)

# Cloning the repository
# git clone https://github.com/Joel-Kazadi/Linear-Regression-using-Python.git

# Changing to the project directory
# cd 'E:/IU/Semester 2/Python programming/Written Assignment/My Work/'

# Switching to the develop branch
# git checkout develop

# Creating a branch for the new feature
# git checkout -b feature

# Adding the changes
# git add .

# Commiting the changes
# git commit -m 'Add new function to the project'

# Pushing the new branch to the remote repository
# git push origin feature

# Updating the repository to the newest commit
# git pull

# Merging remote changes in the project directory
# git merge feature
