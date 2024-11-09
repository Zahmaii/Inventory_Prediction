import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
# Load the logo image
logo_path = "Logo.png"  # Replace with the actual filename of your logo image

# Set Up Login Variables
user_credentials = {
    "admin": "password",
}

# Function for checking credentials
def check_credentials(username, password):
    return user_credentials.get(username) == password

# Login Section
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Placeholder for login status
login_placeholder = st.empty()

if not st.session_state.authenticated:
    with login_placeholder.container():
        st.image("westernimage1.png", caption="Welcome to Inventory Management", use_column_width=True)
        st.title("Login")

        # Login Form
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_credentials(username, password):
                st.session_state.authenticated = True
                st.success("Login successful!")
                login_placeholder.empty()  # Remove the login form immediately
            else:
                st.error("Invalid username or password. Please try again.")

# Main Content
if st.session_state.authenticated:
    # Set Up Menu in Sidebar
    with st.sidebar:
        logo_image = st.image(logo_path, width=100)  # Display logo in the sidebar

        selected = option_menu(
            menu_title="Menu",  
            options=["Data", "Analytics", "Calculation", "Orders", "Log Out"],  
            icons=["clipboard-data", "graph-up-arrow", "calculator-fill", "cart3", "box-arrow-right"],  
            menu_icon="cast",
            default_index=0,  
            orientation="vertical"
        )
        
        # Handle the selected menu item
    # Handling Logout Immediately
    if selected == "Log Out":
        # Clear the authentication status
        st.session_state.authenticated = False
        # Display a success message
        st.success("You have been logged out.")
        # Immediately rerun the script to reflect the logout status
        st.rerun()
        
    # Data Page Content
    if selected == "Data":
        st.image("westernimage1.png", caption="Inventory Management for IZI Restaurant", use_column_width=True)
        st.title("Data Page")

        # Selectbox Inputs for Data Page
        item = st.text_input("Enter the Item", key="item_selection")

        # Sidebar for Number of Days to Enter Data
        num_days = st.sidebar.number_input("Enter Number of Days", min_value=1, max_value=30, value=7, step=1, key="num_days")
        st.session_state.saved_num_days = num_days

        # Sidebar for Price per Kg Input
        price_per_kg_input = st.number_input(f"Enter Price per Kg for {item}", min_value=0.0, step=0.10, format="%.2f", key="price_per_kg_input")
        
        # Save button to store the user data in session state
        if st.button("Save Item & Pricing"):
            st.success("Item and Price saved successfully!")
            st.session_state.price_per_kg = price_per_kg_input
            st.session_state.item = item

        # Sales Data Input Section
        st.subheader(f"Enter Data (in Kilograms)")

        # Collect sales data for the specified number of days
        user_sales_data = []
        for i in range(num_days):
            day_data = st.number_input(f"Day {i+1}", min_value=0, step=1, key=f"day_{i+1}")
            user_sales_data.append(day_data)

        # Save button to store user data
        if st.button("Save Data"):
            # Create a DataFrame with the entered data
            st.session_state['user_sales_data'] = pd.DataFrame({
                'Day': [f"Day {i+1}" for i in range(num_days)],
                'Kilograms': user_sales_data
            })
            st.success("Data saved successfully!")

    elif selected == "Analytics":
        # Retrieve user sales data
        user_sales_data = st.session_state.get('user_sales_data', None)

        if "user_sales_data" in st.session_state and not st.session_state.user_sales_data.empty:
            item = st.session_state.item
            st.title(f"Analytics Page for {item}")

            # Process user sales data
            original_data = st.session_state.user_sales_data.sort_values('Day').reset_index(drop=True)
            original_data.index = original_data.index + 1
            original_data['Day'] = [f"Day {i + 1}" for i in range(len(original_data))]

            st.subheader("User-Entered Data")
            st.dataframe(original_data.set_index('Day').transpose())  # Display data horizontally

            # Display user-entered sales data chart
            st.subheader("User-Entered Data Chart")
            colors = ['#0000FF', '#FF0000']  # Blue to Red gradient
            cmap = LinearSegmentedColormap.from_list('my_gradient', colors, N=len(original_data))

            plt.figure(figsize=(10, 5))
            plt.bar(original_data['Day'], original_data['Kilograms'], color=cmap(np.linspace(0, 1, len(original_data))))
            plt.title("User-Entered Data")
            plt.xlabel("Days")
            plt.ylabel("Kilograms")
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            st.pyplot(plt)

            # Key statistics
            avg_sales = original_data['Kilograms'].mean()
            max_sales = original_data['Kilograms'].max()
            min_sales = original_data['Kilograms'].min()
            st.subheader("Key Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Kilograms", f"{avg_sales:.2f} Kg")
            col2.metric("Max Kilograms", f"{max_sales:.2f} Kg")
            col3.metric("Min Kilograms", f"{min_sales:.2f} Kg")
            st.subheader("-" * 75)

            # Prepare data for ANN
            data_values = original_data['Kilograms'].values.reshape(-1, 1)
            X = np.arange(len(data_values)).reshape(-1, 1)
            
            # Check for sufficient data
            if len(data_values) < 2:
                st.warning("Not enough data for training. Please enter more data.")
            else:
                # Sidebar for selecting Prediction Model and Forecast Period
                prediction_model = st.sidebar.selectbox(
                    "Select Prediction Model",
                    ["Select...", "Linear Regression", "Artificial Neural Network (ANN)", "Support Vector Regression (SVR)"],
                    index=0
                )
                forecast_period = st.sidebar.number_input(
                    "Enter Forecast Period (Days)",
                    min_value=1,   # Minimum forecast period
                    max_value=365, # Maximum forecast period
                    value=7        # Default forecast period
                )
                
                # Ensure that both model and forecast period are selected
                if forecast_period > 0 and prediction_model != "Select...":
                    forecast_period = int(forecast_period)

                    # Data Scaling
                    scaler = StandardScaler()
                    data_values_scaled = scaler.fit_transform(data_values)

                    # Define features and labels for training
                    X = np.arange(len(data_values)).reshape(-1, 1)
                    y = data_values_scaled
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # Initialize model based on selected prediction model
                    if prediction_model == "Artificial Neural Network (ANN)":
                        # Build and compile ANN model with dropout
                        model = Sequential([
                            Dense(20, activation='relu', input_shape=(X_train.shape[1],)),
                            Dropout(0.2),
                            Dense(10, activation='relu'),
                            Dropout(0.2),
                            Dense(5, activation='relu'),
                            Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])

                        # Forecast future data points
                        future_days = np.array(range(len(data_values), len(data_values) + forecast_period)).reshape(-1, 1)
                        future_predictions_scaled = model.predict(future_days)
                        future_predictions = scaler.inverse_transform(future_predictions_scaled)

                    elif prediction_model == "Linear Regression":
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        future_days = np.array(range(len(data_values), len(data_values) + forecast_period)).reshape(-1, 1)
                        future_predictions_scaled = model.predict(future_days)
                        future_predictions = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1))

                    elif prediction_model == "Support Vector Regression (SVR)":
                        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
                        model.fit(X_train, y_train)
                        future_days = np.array(range(len(data_values), len(data_values) + forecast_period)).reshape(-1, 1)
                        future_predictions_scaled = model.predict(future_days)
                        future_predictions = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1))

                    # Prepare forecast data for display
                    forecast_days = [f"Day {i + 1}" for i in range(forecast_period)]
                    forecast_data = pd.DataFrame({'Day': forecast_days, 'Kilograms': future_predictions.flatten()})
                    forecast_data.index = forecast_data.index + 1
                    forecast_data['Kilograms'] = forecast_data['Kilograms'].round(2)

                    # Display forecast data and chart
                    st.subheader(f"Prediction Data for the Next {forecast_period} Days ({prediction_model})")
                    st.dataframe(forecast_data.set_index('Day').transpose())
                    
                    # Create and display forecast chart
                    plt.figure(figsize=(10, 5))
                    plt.plot(forecast_data['Day'], forecast_data['Kilograms'], marker='o', color='b', linestyle='-', label="Forecast")
                    plt.title(f"Forecast for the Next {forecast_period} Days ({prediction_model})")
                    plt.xlabel("Days")
                    plt.ylabel("Kilograms")
                    plt.xticks(rotation=45)
                    plt.grid(True)
                    plt.legend()
                    st.pyplot(plt)

                    # Display key statistics for the forecast period
                    avg_sales = forecast_data['Kilograms'].mean()
                    max_sales = forecast_data['Kilograms'].max()
                    min_sales = forecast_data['Kilograms'].min()
                    st.subheader("Key Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average Kilograms", f"{avg_sales:.2f} Kg")
                    col2.metric("Max Kilograms", f"{max_sales:.2f} Kg")
                    col3.metric("Min Kilograms", f"{min_sales:.2f} Kg")

                    # Save forecast data in session state and provide CSV download
                    st.session_state.forecast_data = forecast_data
                    st.session_state.forecast_period = forecast_period
                    st.session_state.prediction_model = prediction_model
                    st.download_button(
                        label="Download Forecast Data as CSV",
                        data=forecast_data.to_csv(index=False),
                        file_name="forecast_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Please select both a Forecast Period and Prediction Model.")
        else:
            st.warning("Please Enter Data on the Data page to view Analytics.")

    elif selected == "Calculation":
        user_sales_data = st.session_state.get('user_sales_data', pd.DataFrame())
        forecast_data = st.session_state.get('forecast_data', pd.DataFrame())
        forecast_period = st.session_state.get('forecast_period', 0)
        prediction_model = st.session_state.get('prediction_model', 0)
        num_days = st.session_state.get('saved_num_days', 7)
        
        if user_sales_data is not None and not user_sales_data.empty:
            item = st.session_state.item
            st.title(f"Calculation Page for {item}")
            # Sidebar Inputs for Calculation Page
            safety_stock = st.sidebar.number_input("Safety Stock (in Kilograms)", min_value=0, value=0)
            current_inventory = st.sidebar.number_input("Current Inventory (in Kilograms)", min_value=0, value=0)
            ordered_inventory_next_week = st.sidebar.number_input("Ordered Inventory (in Kilograms)", min_value=0, value=0)
            
            # Calculate total kilograms needed for the next week
            total_kilograms_needed = user_sales_data['Kilograms'].sum()
            
            # Display total kilograms from user-entered data if available
            if forecast_data is not None and not forecast_data.empty:
                total_kilograms_forecast = forecast_data['Kilograms'].sum()
            else:
                total_kilograms_forecast = 0
            
            # Display key statistics for the entered and forecasted data
            col7, col8 = st.columns(2)
            col7.markdown(f"**Total Kilograms (Entered Data)**  \n({num_days} Days)")
            col7.metric(" ", f"{total_kilograms_needed:.0f} Kg")
            col8.markdown(f"**Total Kilograms (Prediction Data)**  \n{prediction_model} ({forecast_period} Days)")
            col8.metric(" ", f"{total_kilograms_forecast:.0f} Kg")
            # Separate Sections
            st.subheader("-" * 75)   
            
            # Total Calculation
            total_stock = (total_kilograms_needed + safety_stock) - current_inventory
            # Calculate recommended inventory
            final_inventory = total_stock
            
            # Display safety, current and final inventory
            col9, col10, col11 = st.columns(3)
            col9.metric("**Safety Inventory**", f"{safety_stock:.0f} Kg")
            col10.metric("**Current Inventory**", f"{current_inventory:.0f} Kg")
            col11.metric("**Final Inventory**", f"{final_inventory:.0f} Kg")
            # Separate Sections
            st.subheader("-" * 75)
            
            # Calculate recommended inventory
            recommended_inventory = total_stock + 10
            # Calculate recommended order based on the user input for ordered inventory for next week
            order_inventory = ordered_inventory_next_week  # Adjust based on next week's inventory
            order_inventory = max(order_inventory, 0)  # Ensure no negative order quantity
            
            # Display recommended and ordered inventory
            col12, col13 = st.columns(2)
            col12.metric("**Recommended Inventory**", f"{recommended_inventory:.0f} Kg")
            col13.metric("**Ordered Inventory**", f"{order_inventory:.0f} Kg")

            # Display warning or success message based on inventory prediction
            if total_stock >= order_inventory:
                st.warning(f"You need to order more {item}!")
            else:
                st.success(f"You have enough {item}!")

            st.session_state.final_inventory = final_inventory
            st.session_state.item = item
            st.session_state.recommended_inventory = recommended_inventory
            st.session_state.order_inventory = order_inventory
            st.session_state.total_kilograms_forecast = total_kilograms_forecast
        else:
            st.warning("Please Enter Data on the Data page to view Calculations.")

    elif selected == "Orders":
        # Retrieve item and price per kg from the session state
        price_per_kg = st.session_state.get('price_per_kg', 0)
        final_inventory = st.session_state.get('final_inventory', 0)
        recommended_inventory = st.session_state.get('recommended_inventory', 0)
        order_inventory = st.session_state.get('order_inventory', 0)
        user_sales_data = st.session_state.get('user_sales_data', None)
        forecast_period = st.session_state.get('forecast_period', 0)
        total_kilograms_forecast = st.session_state.get('total_kilograms_forecast', None)

        if user_sales_data is not None and not user_sales_data.empty:
            item = st.session_state.item
            st.title(f"Ordering Page")
            # Display item and price per kg
            col14, col15 = st.columns(2)
            col14.metric("**Item**", f"{item}")
            col15.metric("**Price per Kg**", f"${price_per_kg:.2f}")
            # Separate Sections
            st.subheader("-" * 75)
            
            # Display key statistics for the entered and forecasted data
            col16, col17, col18 = st.columns(3)
            col16.metric("**Total Inventory**", f"{final_inventory:.0f} Kg")
            col17.metric("**Recommended Inventory**", f"{recommended_inventory:.0f} Kg")
            col18.metric(f"**Prediction Inventory** ({forecast_period} Days)", f"{total_kilograms_forecast:.0f} Kg")
            
            # Create a form for placing orders
            with st.form("order_form"):
                st.subheader("Place Your Order")

                # Set recommended_order as the default value for quantity
                quantity = st.number_input(f"Ordered Inventory (in Kilograms)", min_value=0, step=1, value=order_inventory, key='order_quantity')

                # Check if quantity has changed to clear the current order summary
                if 'last_quantity' in st.session_state and st.session_state.last_quantity != quantity:
                    st.session_state.order_summary = None
                st.session_state.last_quantity = quantity

                # Submit button for the form
                submitted = st.form_submit_button("Place Order")

                if submitted:
                    # Calculate total cost for the order
                    total_cost = quantity * price_per_kg
                    
                    # Display order summary if a valid quantity is entered
                    if quantity > final_inventory:
                        order_summary = (
                            f"**Total Cost: ${total_cost:.2f} For {quantity} Kg of {item}**"
                        )
                        st.session_state.order_summary = order_summary
                        st.success("Your order has been placed successfully!")
                    elif quantity <= final_inventory:
                        st.warning("Insufficient inventory to place the order.")
                    else:
                        st.warning("Please enter a quantity greater than 0 to place an order.")

                # Display the latest order summary if it exists
                if st.session_state.get("order_summary"):
                    st.subheader("Order Summary")
                    st.markdown(st.session_state.order_summary)

            # Option to view previous orders
            st.subheader("Previous Orders")
            previous_orders = st.session_state.get('previous_orders', [])

            if previous_orders:
                for order in previous_orders:
                    st.write(order)
            else:
                st.write("No previous orders found.")

            # Button to save current order to previous orders
            if st.button("Save Order to History") and st.session_state.get("order_summary"):
                previous_orders.append(st.session_state.order_summary)
                st.session_state.previous_orders = previous_orders
                st.success("Order saved to history!")

            # Button to delete previous orders
            if st.button("Delete Previous Orders"):
                st.session_state.previous_orders = []  # Clear the previous orders
                st.success("All previous orders have been deleted.")
                
            # Refresh button
            if st.sidebar.button("Refresh Page"):
                st.rerun()  # This will refresh the page and retain the state
                
        else:
            st.warning("Please Enter Data on the Data page to view Orders.")
    # End of Code