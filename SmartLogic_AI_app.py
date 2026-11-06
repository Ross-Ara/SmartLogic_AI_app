
from sklearn import pipeline
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SmartLogix AI",
    page_icon="🚚",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Cleaned datasets"
DATA_DIR = BASE_DIR / "Cleaned datasets"

# Load saved models safely
@st.cache_resource
def load_model(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        return None

    return joblib.load(path)


# Sidebar navigation
st.sidebar.title("🚚 SmartLogix AI")

page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "ETA Prediction",
        "Transportation Mode Prediction",
        "Predictive Maintenance",
        "Route Analytics",
        "Customer Sentiment Analysis"
    ]
)

# HOME
if page == "Home":

    st.title("🚚 SmartLogix AI")

    st.subheader(
        "Intelligent Multi-Modal Logistics "
        "& Autonomous Delivery Platform"
    )

    st.write(
        "An intelligent logistics analytics platform "
        "using Machine Learning and NLP."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Machine Learning Modules", "3")
    col2.metric("NLP Modules", "1")
    col3.metric("Route Analytics Modules", "1")

    st.info("Select a module from the sidebar to begin.")


# ETA PREDICTION


# ==========================================================
# ETA PREDICTION
# ==========================================================

elif page == "ETA Prediction":

    st.title("⏱️ ETA Prediction")
    st.write("Predict delivery time using the trained Machine Learning model.")

    # ------------------------------------------------------
    # 1. LOAD MODEL AND PREPROCESSING FILES
    # ------------------------------------------------------

    model = load_model("eta_random_forest.pkl")
    preprocessor = load_model("eta_preprocessor.pkl")
    feature_columns = load_model("eta_feature_columns.pkl")

    if model is None or preprocessor is None or feature_columns is None:
        st.error(
            "ETA model or preprocessing files not found. "
            "Check the Cleaned datasets folder."
        )

    else:
        st.success("ETA model and preprocessing files loaded successfully!")

        feature_columns = list(feature_columns)

        # --------------------------------------------------
        # 2. LOAD HISTORICAL DELIVERY DATA
        # --------------------------------------------------

        dataset_path = DATA_DIR / "smartlogix_integrated.csv"

        if dataset_path.exists():
            eta_data = pd.read_csv(dataset_path)
            st.caption(
                f"Historical delivery dataset loaded: "
                f"{len(eta_data):,} records"
            )
        else:
            eta_data = pd.DataFrame()
            st.warning("Integrated delivery dataset not found.")

        # --------------------------------------------------
        # 3. GET FEATURE TYPES FROM TRAINED PREPROCESSOR
        # --------------------------------------------------

        numeric_features = []
        categorical_features = []

        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue

            if name == "num":
                numeric_features.extend(list(columns))

            elif name == "cat":
                categorical_features.extend(list(columns))

        # --------------------------------------------------
        # 4. INTERACTIVE ETA PREDICTION FORM
        # --------------------------------------------------

        st.subheader("Enter Delivery Details")

        st.info(
            "Enter delivery information below to generate "
            "an estimated delivery time."
        )

        with st.form("eta_prediction_form"):

            input_values = {}

            col1, col2 = st.columns(2)

            for index, feature in enumerate(feature_columns):

                column = col1 if index % 2 == 0 else col2

                with column:

                    label = feature.replace("_", " ").title()

                    if feature in numeric_features:

                        default_value = 0.0

                        if (
                            not eta_data.empty
                            and feature in eta_data.columns
                        ):
                            numeric_series = pd.to_numeric(
                                eta_data[feature],
                                errors="coerce"
                            ).dropna()

                            if not numeric_series.empty:
                                default_value = float(
                                    numeric_series.median()
                                )

                        input_values[feature] = st.number_input(
                            label,
                            value=default_value,
                            key=f"eta_{feature}"
                        )

                    else:

                        options = []

                        if (
                            not eta_data.empty
                            and feature in eta_data.columns
                        ):
                            options = (
                                eta_data[feature]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )

                        if options:
                            input_values[feature] = st.selectbox(
                                label,
                                options=sorted(options),
                                key=f"eta_{feature}"
                            )
                        else:
                            input_values[feature] = st.text_input(
                                label,
                                value="",
                                key=f"eta_{feature}"
                            )

            predict_button = st.form_submit_button(
                "Predict ETA",
                type="primary"
            )

        # --------------------------------------------------
        # 5. GENERATE PREDICTION FROM FORM
        # --------------------------------------------------

        if predict_button:

            try:
                input_df = pd.DataFrame(
                    [input_values],
                    columns=feature_columns
                )

                processed_input = preprocessor.transform(input_df)

                prediction = float(
                    model.predict(processed_input)[0]
                )

                if prediction >= 0:
                    st.success(
                        f"Predicted Delivery Time: "
                        f"{prediction:.2f} hours"
                    )

                    st.metric(
                        "Estimated Delivery Time",
                        f"{prediction:.2f} hours"
                    )

                    st.write("Input Delivery Details:")
                    st.dataframe(input_df.T)

                else:
                    st.warning(
                        "The model returned a negative ETA. "
                        "Please review the input values."
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # --------------------------------------------------
        # 6. PREDICTION USING REAL HISTORICAL DATA
        # --------------------------------------------------

        st.divider()

        st.subheader("ETA Prediction Using Historical Dataset")

        st.write(
            "Select an existing delivery record from the "
            "integrated dataset and predict its delivery time."
        )

        if eta_data.empty:

            st.warning("Historical delivery data is unavailable.")

        else:

            # Select only records with usable feature columns
            sample_data = eta_data.reindex(
                columns=feature_columns
            )

            max_rows = min(len(sample_data), 100)

            sample_index = st.selectbox(
                "Select Delivery Record",
                options=list(range(max_rows)),
                format_func=lambda x: f"Delivery Record {x + 1}"
            )

            selected_sample = sample_data.iloc[
                [sample_index]
            ].copy()

            st.write("Selected Delivery Details:")
            st.dataframe(selected_sample.T.astype(str), use_container_width=True)

            if st.button(
                "Predict ETA for Selected Delivery",
                type="primary"
            ):

                try:

                    processed_sample = preprocessor.transform(
                        selected_sample
                    )

                    sample_prediction = float(
                        model.predict(processed_sample)[0]
                    )

                    st.success(
                        f"Predicted Delivery Time: "
                        f"{sample_prediction:.2f} hours"
                    )

                    st.metric(
                        "Predicted ETA",
                        f"{sample_prediction:.2f} hours"
                    )

                    # Compare prediction with actual delivery time
                    if "actual_delivery_hours" in eta_data.columns:

                        actual_value = pd.to_numeric(
                            pd.Series(
                                [eta_data.iloc[sample_index][
                                    "actual_delivery_hours"
                                ]]
                            ),
                            errors="coerce"
                        ).iloc[0]

                        if pd.notna(actual_value):

                            actual_value = float(actual_value)

                            absolute_error = abs(
                                actual_value - sample_prediction
                            )

                            result_col1, result_col2 = st.columns(2)

                            with result_col1:
                                st.metric(
                                    "Actual Delivery Time",
                                    f"{actual_value:.2f} hours"
                                )

                            with result_col2:
                                st.metric(
                                    "Absolute Prediction Error",
                                    f"{absolute_error:.2f} hours"
                                )

                            comparison_df = pd.DataFrame({
                                "Delivery Time": [
                                    sample_prediction,
                                    actual_value
                                ]
                            }, index=["Predicted", "Actual"])

                            st.bar_chart(comparison_df)

                except Exception as e:
                    st.error(
                        f"Sample prediction failed: {e}"
                    )

    # ==========================================
    # OVERALL ETA MODEL PERFORMANCE
    # ==========================================

    st.divider()
    st.subheader("📊 Overall ETA Model Performance")

    metrics_path = DATA_DIR / "eta_final_model_comparison.csv"

    if metrics_path.exists():

        metrics_df = pd.read_csv(metrics_path)

        # Display all trained model results
        st.write("Model Performance Comparison")
        st.dataframe(metrics_df, use_container_width=True)

        # Standardize column names
        metrics_df.columns = [
            str(col).strip().lower().replace(" ", "").replace("_", "")
            for col in metrics_df.columns
        ]

        # Identify metric columns
        mae_col = "mae"
        rmse_col = "rmse"
        r2_col = "r2" if "r2" in metrics_df.columns else "r2score"

        if all(
            col in metrics_df.columns
            for col in [mae_col, rmse_col, r2_col]
        ):

            # Select best model based on lowest MAE
            metrics_df[mae_col] = pd.to_numeric(
                metrics_df[mae_col], errors="coerce"
            )

            valid_metrics = metrics_df.dropna(subset=[mae_col])

            if not valid_metrics.empty:

                best_model = valid_metrics.loc[
                    valid_metrics[mae_col].idxmin()
                ]

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "MAE",
                    f"{float(best_model[mae_col]):.2f} hours"
                )

                col2.metric(
                    "RMSE",
                    f"{float(best_model[rmse_col]):.2f} hours"
                )

                col3.metric(
                    "R² Score",
                    f"{float(best_model[r2_col]):.4f}"
                )

                st.info(
                    "MAE and RMSE: Lower values are better. "
                    "R²: Higher values are better."
                )

            else:
                st.warning("No valid MAE values found.")

        else:
            st.warning("Required metric columns were not found.")
            st.write("Available columns:", metrics_df.columns.tolist())

    else:
        st.warning("ETA final model comparison CSV not found.")
        
# TRANSPORTATION MODE
elif page == "Transportation Mode Prediction":


    st.title("🚚 Transportation Mode Prediction")
    st.write(
    "Predict the transportation mode using the trained "
    "Decision Tree classification model."
)

    # Load trained model and preprocessing files
    transport_model = load_model("transport_best_model.pkl")
    transport_preprocessor = load_model("transport_preprocessor.pkl")
    transport_features = load_model("transport_feature_columns.pkl")

    if (
        transport_model is None
        or transport_preprocessor is None
        or transport_features is None
    ):
        st.error("Transportation model or preprocessing files not found.")

    else:
        st.success("Transportation prediction model loaded successfully!")

        # Load historical dataset for form options
        transport_data_path = DATA_DIR / "smartlogix_transport_cleaned.csv"

        if transport_data_path.exists():
            transport_data = pd.read_csv(transport_data_path)
        else:
            transport_data = pd.DataFrame(columns=transport_features)

        st.subheader("Enter Delivery Details")

        def transport_options(column):
            if column in transport_data.columns:
                values = (
                    transport_data[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                return sorted(values) if values else ["Unknown"]
            return ["Unknown"]

        with st.form("transport_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                quantity = st.number_input(
                    "Quantity", min_value=1, value=3, step=1
                )

                origin_hub = st.selectbox(
                    "Origin Hub", transport_options("origin_hub")
                )

                destination_city = st.selectbox(
                    "Destination City",
                    transport_options("destination_city")
                )

            with col2:
                distance_km = st.number_input(
                    "Distance (km)",
                    min_value=0.1,
                    value=100.0,
                    step=1.0
                )

                origin_city = st.selectbox(
                    "Origin City", transport_options("origin_city")
                )

                destination_state = st.selectbox(
                    "Destination State",
                    transport_options("destination_state")
                )

            predict_transport = st.form_submit_button(
                "Predict Transportation Mode",
                type="primary"
            )

        if predict_transport:

            input_data = pd.DataFrame([{
                "quantity": quantity,
                "distance_km": distance_km,
                "origin_hub": origin_hub,
                "origin_city": origin_city,
                "destination_city": destination_city,
                "destination_state": destination_state
            }])

            try:
                # Ensure feature order matches training
                input_data = input_data.reindex(columns=transport_features)

                processed_input = transport_preprocessor.transform(input_data)

                prediction = transport_model.predict(processed_input)[0]

                st.success(
                    f"🚚 Predicted Transportation Mode: {prediction}"
                )

                st.subheader("Input Delivery Details")
                st.dataframe(input_data.astype(str), use_container_width=True)

                if hasattr(transport_model, "predict_proba"):
                    probabilities = transport_model.predict_proba(
                        processed_input
                    )[0]

                    confidence = float(max(probabilities)) * 100

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

            except Exception as e:
                st.error(f"Transportation prediction failed: {e}")

        # ============================================
    # TRANSPORTATION MODE - HISTORICAL PREDICTION
    # ============================================

    st.divider()
    st.subheader("📊 Transportation Prediction Using Historical Dataset")

    st.write(
        "Select an existing delivery record and compare "
        "the predicted transportation mode with the actual mode."
    )

    dataset_path = DATA_DIR / "smartlogix_integrated.csv"

    if dataset_path.exists():

        transport_data = pd.read_csv(dataset_path)

        required_columns = [
            "quantity",
            "distance_km",
            "origin_hub",
            "origin_city",
            "destination_city",
            "destination_state"
        ]

        if all(col in transport_data.columns for col in required_columns):

            transport_data = transport_data.dropna(
                subset=["transport_mode"]
            ).reset_index(drop=True)

            if transport_data.empty:
                st.warning("No transportation records available.")

            else:
                max_rows = min(100, len(transport_data))

                selected_index = st.selectbox(
                    "Select Delivery Record",
                    options=list(range(max_rows)),
                    format_func=lambda x: f"Delivery Record {x + 1}",
                    key="transport_historical_record"
                )

                selected_record = transport_data.iloc[
                    [selected_index]
                ].copy()

                st.write("Selected Delivery Details:")

                display_columns = required_columns + ["transport_mode"]

                st.dataframe(
                    selected_record[display_columns].astype(str),
                    use_container_width=True
                )

                if st.button(
                    "Predict Mode for Selected Delivery",
                    type="primary"
                ):

                    try:
                        transport_features = load_model(
                            "transport_feature_columns.pkl"
                        )

                        transport_preprocessor = load_model(
                            "transport_preprocessor.pkl"
                        )

                        transport_model = load_model(
                            "transport_best_model.pkl"
                        )

                        if any(
                            item is None
                            for item in [
                                transport_features,
                                transport_preprocessor,
                                transport_model
                            ]
                        ):
                            st.error("Required transportation model files not found.")

                        else:
                            input_data = selected_record[
                                transport_features
                            ].copy()

                            processed_input = (
                                transport_preprocessor.transform(input_data)
                            )

                            predicted_mode = transport_model.predict(
                                processed_input
                            )[0]

                            actual_mode = selected_record[
                                "transport_mode"
                            ].iloc[0]

                            col1, col2 = st.columns(2)

                            with col1:
                            
                                 st.metric(
                                    "Actual Transportation Mode",
                                str(actual_mode).strip().title()
                                    )

                            with col2:
                                st.metric(
                                    "Predicted Transportation Mode",
                                    str(predicted_mode).strip().title()
                                    )

                            if str(actual_mode).strip().casefold() == str(predicted_mode).strip().casefold():
                                st.success(
                                    "✅ Prediction matches the actual transportation mode!"
                                )
                            else:
                                st.warning(
                                    "⚠️ Prediction differs from the actual transportation mode."
                                )

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        else:
            st.warning("Required transportation columns are missing.")

    else:
        st.warning("Integrated historical dataset not found.")

        # ============================================
    # TRANSPORTATION MODEL PERFORMANCE METRICS
    # ============================================

    st.divider()
    st.subheader("📊 Overall Transportation Model Performance")

    metrics_path = DATA_DIR / "transport_model_comparison.csv"

    if metrics_path.exists():

        metrics_df = pd.read_csv(metrics_path)

        st.write("Model Performance Comparison")

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        # Identify metric columns
        metric_columns = {
            "Accuracy": "Accuracy",
            "Precision": "Precision",
            "Recall": "Recall",
            "F1 Score": "F1 Score"
        }

        # Select the model actually saved as the best model
        model_info = load_model("transport_model_info.pkl")

        selected_model_name = (
            model_info.get("model_name")
            if isinstance(model_info, dict)
            else None
        )

        if "Model" in metrics_df.columns and selected_model_name:
            selected_rows = metrics_df[
                metrics_df["Model"].astype(str).str.casefold()
                == str(selected_model_name).casefold()
            ]
        else:
            selected_rows = pd.DataFrame()

        if not selected_rows.empty:

            best_row = selected_rows.iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            for column, (label, metric_name) in zip(
                [col1, col2, col3, col4],
                metric_columns.items()
            ):
                if metric_name in best_row.index:
                    with column:
                        st.metric(
                            label,
                            f"{float(best_row[metric_name]):.2%}"
                        )

            st.info(
                f"Selected Model: {selected_model_name}"
            )

        else:
            st.warning(
                "The selected model was not found in the comparison CSV. "
                "Check the model names and column headings."
            )

    else:
        st.warning(
            "transport_model_comparison.csv not found in Cleaned datasets."
        )



# PREDICTIVE MAINTENANCE

elif page == "Predictive Maintenance":

    st.title("🔧 Predictive Maintenance")
    st.write(
        "Predict whether a fleet vehicle may require maintenance "
        "within the next 30 days using a Random Forest model."
    )

    maintenance_model = load_model("maintenance_random_forest.pkl")
    maintenance_preprocessor = load_model("maintenance_preprocessor.pkl")
    maintenance_features = load_model("maintenance_feature_columns.pkl")

    if (
        maintenance_model is None
        or maintenance_preprocessor is None
        or maintenance_features is None
    ):
        st.error("Maintenance model or preprocessing files not found.")

    else:
        st.success("Predictive Maintenance model loaded successfully!")

        st.subheader("Enter Vehicle Details")

        # Use the categories learned by the saved preprocessor.
        vehicle_options = None

        for name, transformer, columns in maintenance_preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue

            if "vehicle_type" in list(columns):
                encoder = (
                    transformer.named_steps.get("encoder")
                    if hasattr(transformer, "named_steps")
                    else transformer
                )

                if hasattr(encoder, "categories_"):
                    position = list(columns).index("vehicle_type")
                    vehicle_options = [
                        str(value)
                        for value in encoder.categories_[position]
                    ]

        if not vehicle_options:
            st.error(
                "Unable to determine vehicle types from the saved "
                "preprocessor. Please inspect its categories."
            )
        else:
            with st.form("maintenance_prediction_form"):

                vehicle_type = st.selectbox(
                    "Vehicle Type",
                    vehicle_options
                )

                previous_maintenance_count = st.number_input(
                    "Previous Maintenance Count",
                    min_value=0,
                    value=2,
                    step=1
                )

                days_since_previous_maintenance = st.number_input(
                    "Days Since Previous Maintenance",
                    min_value=0,
                    value=90,
                    step=1
                )

                predict_maintenance = st.form_submit_button(
                    "Predict Maintenance Requirement",
                    type="primary"
                )

            if predict_maintenance:

                input_data = pd.DataFrame([{
                    "vehicle_type": vehicle_type,
                    "previous_maintenance_count": previous_maintenance_count,
                    "days_since_previous_maintenance":
                        days_since_previous_maintenance
                }])

                input_data = input_data.reindex(
                    columns=maintenance_features
                )

                try:
                    processed_input = maintenance_preprocessor.transform(
                        input_data
                    )

                    prediction = maintenance_model.predict(
                        processed_input
                    )[0]

                    if int(prediction) == 1:
                        st.warning(
                            "⚠️ Maintenance may be required within 30 days."
                        )
                    else:
                        st.success(
                            "✅ No maintenance requirement predicted "
                            "within 30 days."
                        )

                    st.subheader("Input Vehicle Details")
                    st.dataframe(
                        input_data.astype(str),
                        use_container_width=True
                    )

                    if hasattr(maintenance_model, "predict_proba"):
                        probabilities = maintenance_model.predict_proba(
                            processed_input
                        )[0]

                        classes = list(maintenance_model.classes_)

                        if 1 in classes:
                            maintenance_probability = probabilities[
                                classes.index(1)
                            ]

                            st.metric(
                                "Estimated Maintenance Probability",
                                f"{maintenance_probability * 100:.2f}%"
                            )

                    st.caption(
                        "Prototype prediction based on historical data. "
                        "Not a substitute for vehicle inspections or "
                        "scheduled maintenance."
                    )

                except Exception as e:
                    st.error(f"Maintenance prediction failed: {e}")

        st.divider()
        st.subheader("📊 Overall Maintenance Model Performance")

        comparison_path = (
            DATA_DIR / "maintenance_model_comparison.csv"
        )

        if comparison_path.exists():

            maintenance_results = pd.read_csv(comparison_path)

            st.dataframe(
                maintenance_results,
                use_container_width=True,
                hide_index=True
            )

            selected_result = maintenance_results[
                maintenance_results["Model"] == "Random Forest"
            ]

            if not selected_result.empty:

                result = selected_result.iloc[0]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Accuracy",
                    f"{result['Accuracy'] * 100:.2f}%"
                )

                col2.metric(
                    "Precision",
                    f"{result['Precision'] * 100:.2f}%"
                )

                col3.metric(
                    "Recall",
                    f"{result['Recall'] * 100:.2f}%"
                )

                col4.metric(
                    "F1 Score",
                    f"{result['F1 Score'] * 100:.2f}%"
                )

            st.info("Selected Model: Random Forest")

        else:
            st.warning("Maintenance model comparison CSV not found.")



# ROUTE ANALYTICS
elif page == "Route Analytics":

    st.title("🗺️ Route Analytics")
    st.write(
        "Analyze historical delivery routes, travel times, "
        "distances, and route performance."
    )

    route_file = DATA_DIR / "route_performance_summary.csv"

    if route_file.exists():

        route_df = pd.read_csv(route_file)

        # Convert numeric columns safely
        numeric_columns = [
            "average_travel_time_min",
            "average_distance_km",
            "number_of_trips"
        ]

        for col in numeric_columns:
            route_df[col] = pd.to_numeric(
                route_df[col], errors="coerce"
            )

        st.success("Route performance dataset loaded successfully!")

        # Overall route metrics
        st.subheader("📊 Overall Route Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Routes",
                f"{route_df['route_id'].nunique():,}"
            )

        with col2:
            st.metric(
                "Average Travel Time",
                f"{route_df['average_travel_time_min'].mean():.2f} min"
            )

        with col3:
            st.metric(
                "Average Route Distance",
                f"{route_df['average_distance_km'].mean():.2f} km"
            )

        with col4:
            st.metric(
                "Total Recorded Trips",
                f"{route_df['number_of_trips'].sum():,.0f}"
            )

        st.divider()

        # Fastest routes
        st.subheader("🚀 Top 10 Fastest Routes")

        fastest_routes = route_df.dropna(
            subset=["average_travel_time_min"]
        ).nsmallest(10, "average_travel_time_min")

        st.dataframe(
            fastest_routes,
            use_container_width=True
        )

        st.bar_chart(
            fastest_routes.set_index("route_id")[
                "average_travel_time_min"
            ]
        )

        st.divider()

        # Slowest routes
        st.subheader("🐢 Top 10 Slowest Routes")

        slowest_routes = route_df.dropna(
            subset=["average_travel_time_min"]
        ).nlargest(10, "average_travel_time_min")

        st.dataframe(
            slowest_routes,
            use_container_width=True
        )

        st.bar_chart(
            slowest_routes.set_index("route_id")[
                "average_travel_time_min"
            ]
        )

        st.divider()

        # Route distance analysis
        st.subheader("📏 Route Distance Analysis")

        distance_df = route_df.dropna(
            subset=["average_distance_km"]
        )

        st.write(
            f"Routes with valid distance data: {len(distance_df):,}"
        )

        st.bar_chart(
            distance_df.nlargest(
                10, "average_distance_km"
            ).set_index("route_id")[
                "average_distance_km"
            ]
        )

    else:
        st.error(
            "route_performance_summary.csv not found "
            "inside Cleaned datasets."
        )

    # ==========================================
    # STEP 4.4 - TRAFFIC CONGESTION ANALYSIS
    # ==========================================

    st.divider()
    st.subheader("🚦 Traffic Congestion Analysis")

    traffic_file = DATA_DIR / "traffic_clean.csv"

    if traffic_file.exists():

        traffic_df = pd.read_csv(traffic_file)

        
        # Convert congestion values to numeric
        traffic_df["congestion_index"] = pd.to_numeric(
            traffic_df["congestion_index"],
            errors="coerce"
        )

        # Normalize mixed decimal and percentage values
        # Assumption: values above 1 represent percentages
        traffic_df["congestion_index"] = (
            traffic_df["congestion_index"].where(
                traffic_df["congestion_index"] <= 1,
                traffic_df["congestion_index"] / 100
            )
        )

        st.caption(
            "Congestion values above 1 are assumed to be "
            "percentages and converted to a 0–1 scale."
        )

        traffic_df["avg_speed_kmph"] = pd.to_numeric(
            traffic_df["avg_speed_kmph"], errors="coerce"
        )

        traffic_df["hour_of_day"] = pd.to_numeric(
            traffic_df["hour_of_day"], errors="coerce"
        )

        st.success("Traffic dataset loaded successfully!")

        # Traffic KPIs
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Traffic Records",
                f"{len(traffic_df):,}"
            )

        with col2:
            st.metric(
                "Average Congestion Index",
                f"{traffic_df['congestion_index'].mean():.3f}"
            )

        with col3:
            st.metric(
                "Average Vehicle Speed",
                f"{traffic_df['avg_speed_kmph'].mean():.2f} km/h"
            )

        st.divider()

        # Average congestion by hour
        st.subheader("⏰ Hourly Traffic Congestion")

        hourly_traffic = (
            traffic_df.groupby("hour_of_day")[
                "congestion_index"
            ]
            .mean()
            .sort_index()
        )

        st.line_chart(hourly_traffic)

        # Average speed by hour
        st.subheader("🚗 Average Speed by Hour")

        hourly_speed = (
            traffic_df.groupby("hour_of_day")[
                "avg_speed_kmph"
            ]
            .mean()
            .sort_index()
        )

        st.line_chart(hourly_speed)

        st.divider()

        # Congestion by city
        st.subheader("🏙️ Top 10 Cities by Average Congestion")

        city_congestion = (
            traffic_df.groupby("city")[
                "congestion_index"
            ]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(city_congestion)

        # Traffic records
        with st.expander("View Traffic Dataset"):
            st.dataframe(
                traffic_df.head(100),
                use_container_width=True
            )

    else:
        st.warning(
            "traffic_clean.csv not found in Cleaned datasets."
        )

    
# STEP 4.7 - Traffic Insights and Recommendations

    st.divider()
    st.subheader("🚦 Traffic Insights & Delivery Recommendations")

    traffic_insights = traffic_df.copy()

    traffic_insights["hour_of_day"] = pd.to_numeric(
        traffic_insights["hour_of_day"], errors="coerce"
    )

    traffic_insights["avg_speed_kmph"] = pd.to_numeric(
        traffic_insights["avg_speed_kmph"], errors="coerce"
    )

    hourly_summary = (
        traffic_insights.dropna(
            subset=["hour_of_day", "congestion_index"]
        )
        .groupby("hour_of_day")
        .agg(
            avg_congestion=("congestion_index", "mean"),
            avg_speed=("avg_speed_kmph", "mean"),
            records=("congestion_index", "size")
        )
        .reset_index()
    )

    if not hourly_summary.empty:
        least_congested = hourly_summary.loc[
            hourly_summary["avg_congestion"].idxmin()
        ]

        most_congested = hourly_summary.loc[
            hourly_summary["avg_congestion"].idxmax()
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Least Congested Recorded Hour",
                f"{int(least_congested['hour_of_day']):02d}:00"
            )

        with col2:
            st.metric(
                "Most Congested Recorded Hour",
                f"{int(most_congested['hour_of_day']):02d}:00"
            )

        st.info(
            "📦 Historical delivery planning suggestion: "
            f"Consider departures around "
            f"{int(least_congested['hour_of_day']):02d}:00 "
            "when other operational conditions permit."
        )

        st.warning(
            "🚚 Potential delay period: "
            f"Traffic congestion was highest around "
            f"{int(most_congested['hour_of_day']):02d}:00. "
            "Allow additional travel time when planning deliveries."
        )

        st.caption(
            "Recommendations are based on historical average "
            "congestion, not real-time traffic or optimized routing."
        )

        st.subheader("📊 Hourly Traffic Planning Summary")
        st.dataframe(
            hourly_summary.round(3),
            use_container_width=True
        )
    else:
        st.warning("Insufficient data for hourly traffic insights.")


# CUSTOMER SENTIMENT ANALYSIS
elif page == "Customer Sentiment Analysis":

    
        st.title("💬 Customer Sentiment Analysis")

        st.write(
        "Analyze customer delivery reviews using "
        "a pretrained Transformer sentiment model."
    )

        @st.cache_resource
        def load_transformer_sentiment():
            from transformers import pipeline

            return pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )

        try:
            sentiment_analyzer = load_transformer_sentiment()
            st.success("Transformer sentiment model loaded successfully!")

            review = st.text_area(
            "Enter Customer Review",
            placeholder="The delivery was fast and excellent!"
            )

            if st.button("Analyze Sentiment", key="transformer_sentiment"):
                if review.strip():
                    result = sentiment_analyzer(
                    review.strip(),
                    truncation=True,
                    max_length=512
                )[0]

                prediction = result["label"].capitalize()
                confidence = result["score"]

                st.subheader("Predicted Sentiment")

                if prediction == "Positive":
                    st.success(f"😊 {prediction}")
                elif prediction == "Negative":
                    st.error(f"😞 {prediction}")
                else:
                    st.info(f"😐 {prediction}")

                st.write(f"Model confidence: {confidence:.2%}")
                st.caption(
                    "Confidence is the model's score, "
                    "not a guarantee of correctness."
                )
            else:
                st.warning("Please enter a customer review.")

        except Exception as e:
            st.error(f"Unable to load sentiment model: {e}")