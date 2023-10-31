import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo de SVM
svc_model = joblib.load('models/svc_clf.pkl')

# Preguntas correspondientes a los atributos
questions = [
    "If one of us apologizes when our discussion deteriorates, the discussion ends.",
    "I know we can ignore our differences, even if things get hard sometimes.",
    "The time I spent with my wife is special for us.",
    "Most of our goals are common to my spouse.",
    "I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.",
    "My spouse and I have similar values in terms of personal freedom.",
    "Our dreams with my spouse are similar and harmonious.",
    "We're compatible with my spouse about what love should be.",
    "I enjoy our holidays with my wife.",
    "I can tell you what kind of stress my spouse is facing in her/his life.",
    "I know my spouse's basic anxieties."
]

# Función para hacer predicciones
def predict_svm(input_data):
    prediction = svc_model.predict(input_data)
    return prediction

# Función principal de la aplicación
def main():
    st.title('Divorce Prediction with SVM')
    st.sidebar.header('User Input')

    # Crear widgets para ingresar datos de entrada
    # Utiliza un bucle para generar widgets para cada pregunta
    input_data = []
    for i, question in enumerate(questions):
        answer = st.sidebar.slider(question, min_value=0, max_value=4)
        input_data.append(answer)

    input_data = pd.DataFrame({f'Atr{i+1}': [value] for i, value in enumerate(input_data)})

    if st.sidebar.button('Predict'):
        prediction = predict_svm(input_data)
        if prediction[0] == 0:
            st.write('Prediction: Divorce')
        else:
            st.write('Prediction: Married')

if __name__ == '__main__':
    main()
