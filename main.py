import streamlit as st
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px
import plotly.graph_objects as go
import time
import math
import random 
# Funciones 
def eGreedyMethod(epsilon, banditMeans, banditVariance, w0,
                  numberSteps, numberEpisodics, numberActions, 
                  estaticFactor = -1):
  # Función que calcula las recompensas promedio luego de usar el método e-greedy para 
  # buscar la mejor acción posible en el problema de multi-bandits. 
  # Los resultados se realizan para varios episodios (numberEpisodics) 
  # en cada episodio se realizan un número de iteraciones o pasos (numberSteps)

  # Parámetros:
  # epsilon: factor del método e-greedy
  # banditEvns: clase que simula el problema de multi-bands
  # banditScenery: elección del escenario del problema 1, 2, 3
  # numberSteps: número de iteraciones en cada episodio
  # numberEpisodics: número de episodios
  # numberActions: número de acciones disponibles para tomar
  # estaticFactor: indica al algoritmo si se quiere trabajar de forma estatica (-1) o no, 
  #                en el último caso se debe incluir un número entre 0 y 1 para ser **

  # Salidas:
  # rewardsMean: promedio de las recompensas en cada iteración para el total de episodios
  # meanCountActions: promedio del número de acciones tomadas en cada episodio

  
  actionsAvaliable = range(numberActions)
  rewardsMean = np.zeros((numberSteps,))
  meanCountActions = np.zeros((numberActions,))

  for j in range(numberEpisodics):
    Q = np.zeros((numberActions))
    rewards = []
    actionsTaken = []
    for i in range(numberSteps):
      actionGreedy = np.argmax(Q)
      probabilitiesAction = (epsilon/numberActions)*np.ones((numberActions))
      probabilitiesAction[actionGreedy] = 1-epsilon + (epsilon/numberActions)
      action = random.choices(actionsAvaliable, weights=probabilitiesAction, k=1)[0]
      actionsTaken.append(action)
      # Generación de bandits con media afectada por np.cos(w0*t)
      reward = np.random.normal(banditMeans[action]+ np.cos(w0*i), banditVariance[action])  
      rewards.append(reward)
      if estaticFactor == -1:
        Q[action] += (1/actionsTaken.count(action))*(reward - Q[action])
      elif 0 <= estaticFactor and estaticFactor <= 1:
        Q[action] += (estaticFactor)*(reward - Q[action])
      else:
        print('Ingreso un valor invalido para estaticFactor')
        print('\nIngrese un número entre 0 y 1, o indique -1 para método estatico')
    rewardsMean = rewardsMean + (1/(j+1))* (rewards - rewardsMean)
    countActions  = np.bincount(actionsTaken, minlength=numberActions)
    meanCountActions = meanCountActions + (1/(j+1))*(countActions-meanCountActions)
  return rewardsMean, meanCountActions*100/numberSteps

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

st.set_page_config(page_title='Bono1', layout="wide")

# Diseño
header = st.container()
st.markdown("""---""")  
context = st.container()
st.markdown("""---""")  
context2 = st.container()
st.markdown("""---""")  


with header:
  st.title("Multi-Armed Bandit no estacionario")
  st.header("Bono 1")
  st.subheader("Interfaz")
  st.subheader("Reinforcement Learning")
  st.write("Estudiante: Manuela Viviana Chacón Chamorro")
  st.markdown("""---""") 
  st.write(r'En esta interfaz se comparan lso resultados del método $\epsilon$-greedy con diferentes valores de $\alpha$ constantes y la configuración $\alpha = 1/t$')

with context:
  flag_train = False
  st.title("Escenario 1")
  st.write(r"Escenario con media $\mu = \beta + \cos(w_0 t)$, donde $w_0  = \pi/100000$")
  with st.form(key='form1'):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
      numberEpisodics = st.number_input("Número de episodios", step=100, format="%d", min_value=100)
    with col_b:
      alpha = st.slider('alpha', 0.001, 1.0, 0.2) 
    with col_c:
      epsilon = st.slider('epsilon', 0.001, 1.0, 0.1)
    submitted = st.form_submit_button('Ejecutar')
    if submitted:
      flag_train = True
      np.random.seed(2023)
      k = 12
      banditVariance = np.ones(k)
      banditMeans = [np.random.normal(0, 1) for _ in range(k)]
      betterAction = np.argmax(banditMeans)
      w0 = math.pi/100000
      rewardsMean, meanCountActions =  eGreedyMethod(epsilon, banditMeans, banditVariance, w0,
                  1000, numberEpisodics, 12, alpha)
      
      figAlpha1, axAlpha1 = plt.subplots(figsize=(10,4))
      axAlpha1.plot(rewardsMean, label=f'Promedio evolución recompensa', color = '#24C0C4')
      axAlpha1.plot([max(banditMeans)+ np.cos(w0*t) for t in range(1000)], label=f'$\mu$', color = 'blue')
      axAlpha1.set_title(f'Evolución de la recompensa promedio en {numberEpisodics} episodios', fontsize = 14)
      axAlpha1.set_ylabel('Recompensa promedio', fontsize = 14)
      axAlpha1.set_xlabel('Iteraciones', fontsize = 14)
      axAlpha1.legend()
      axAlpha1.grid()

      
      figAlphaBar1, axAlphaBar1 = plt.subplots(figsize=(10,4))
      axAlphaBar1.bar(range(12), meanCountActions, color = '#24C0C4')
      axAlphaBar1.set_xticks(np.arange(12), np.arange(12), fontsize = 14)
      axAlphaBar1.set_title(f'Porcentaje promedio de acciones tomadas en todos los episodios por $\epsilon$, \n mejor acción {betterAction}', 
                fontsize = 14)
      axAlphaBar1.set_xlabel('Acciones', fontsize = 14)
      axAlphaBar1.set_ylabel('% de cantidad de veces escogida', fontsize = 14)
      axAlphaBar1.grid(axis='y')
      

      rewardsMean, meanCountActions =  eGreedyMethod(epsilon, banditMeans, banditVariance, w0,
                  1000, numberEpisodics, 12, -1)
      
      figAlpha2, axAlpha2 = plt.subplots(figsize=(10,4))
      axAlpha2.plot(rewardsMean, label='Promedio evolución recompensa', color = '#F63366')
      axAlpha2.plot([max(banditMeans)+ np.cos(w0*t) for t in range(1000)], label=f'$\mu$', color = 'blue')
      axAlpha2.set_title(f'Evolución de la recompensa promedio en {numberEpisodics} episodios', fontsize = 14)
      axAlpha2.set_ylabel('Recompensa promedio', fontsize = 14)
      axAlpha2.set_xlabel('Iteraciones', fontsize = 14)
      axAlpha2.legend()
      axAlpha2.grid()

      figAlphaBar2, axAlphaBar2 = plt.subplots(figsize=(10,4))
      axAlphaBar2.bar(range(12), meanCountActions, color = '#F63366')
      axAlphaBar2.set_xticks(np.arange(12), np.arange(12), fontsize = 14)
      axAlphaBar2.set_title(f'Porcentaje promedio de acciones tomadas en todos los episodios por $\epsilon$, \n mejor acción {betterAction}', 
                fontsize = 14)
      axAlphaBar2.set_xlabel('Acciones', fontsize = 14)
      axAlphaBar2.set_ylabel('% de cantidad de veces escogida', fontsize = 14)
      axAlphaBar2.grid(axis='y')
  
  col_a1, col_b1= st.columns(2)
  with st.container():
    
    with col_a1:
      st.subheader(rf'Resultados con $\alpha$ = {alpha} y $\epsilon$ = {epsilon}')
      if submitted:
        #st.write(f'Evolución de la recompensa promedio en {numberEpisodics} episodios')
        st.pyplot(figAlpha1)

    with col_b1:
      st.subheader(rf'Resultados con $\alpha = 1/t$  y $\epsilon$ = {epsilon}')
      if submitted:
        #st.subheader(f'Evolución de la recompensa promedio en {numberEpisodics} episodios')
        st.pyplot(figAlpha2)
    
  col_bar1, col_bar2 = st.columns(2)
  
  with st.container():
    #st.subheader(fr'Diagrama de barras del porcentaje promedio de acciones tomadas en todos los episodios por $\epsilon$, mejor acción {betterAction}')
    with col_bar1:
      if submitted:
        st.pyplot(figAlphaBar1)
    
    with col_bar2:
      if submitted:
        st.pyplot(figAlphaBar2)


with context2:
  flag_train2 = False
  st.title("Escenario 2")
  st.write(r"Escenario con media $\mu = \beta + \cos(w_0 t)$ donde $w_0  = \pi/1000$")
  with st.form(key='form2'):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
      numberEpisodics = st.number_input("Número de episodios", step=100, format="%d", min_value=100)
    with col_b:
      alpha2 = st.slider('alpha', 0.001, 1.0, 0.2) 
    with col_c:
      epsilon2 = st.slider('epsilon', 0.001, 1.0, 0.1)
    submitted = st.form_submit_button('Ejecutar')
    if submitted:
      flag_train2 = True
      np.random.seed(2023)
      k = 12
      banditVariance = np.ones(k)
      banditMeans = [np.random.normal(0, 1) for _ in range(k)]
      betterAction = np.argmax(banditMeans)
      w0 = math.pi/1000
      rewardsMean, meanCountActions =  eGreedyMethod(epsilon, banditMeans, banditVariance, w0,
                  1000, numberEpisodics, 12, alpha)
      
      figAlpha1, axAlpha1 = plt.subplots(figsize=(10,4))
      axAlpha1.plot(rewardsMean, label=f'Promedio evolución recompensa', color = '#24C0C4')
      axAlpha1.plot([max(banditMeans)+ np.cos(w0*t) for t in range(1000)], label=f'$\mu$', color = 'blue')
      axAlpha1.set_title(f'Evolución de la recompensa promedio en {numberEpisodics} episodios', fontsize = 14)
      axAlpha1.set_ylabel('Recompensa promedio', fontsize = 14)
      axAlpha1.set_xlabel('Iteraciones', fontsize = 14)
      axAlpha1.legend()
      axAlpha1.grid()

      
      figAlphaBar1, axAlphaBar1 = plt.subplots(figsize=(10,4))
      axAlphaBar1.bar(range(12), meanCountActions, color = '#24C0C4')
      axAlphaBar1.set_xticks(np.arange(12), np.arange(12), fontsize = 14)
      axAlphaBar1.set_title(f'Porcentaje promedio de acciones tomadas en todos los episodios por $\epsilon$, \n mejor acción {betterAction}', 
                fontsize = 14)
      axAlphaBar1.set_xlabel('Acciones', fontsize = 14)
      axAlphaBar1.set_ylabel('% de cantidad de veces escogida', fontsize = 14)
      axAlphaBar1.grid(axis='y')
      

      rewardsMean, meanCountActions =  eGreedyMethod(epsilon, banditMeans, banditVariance, w0,
                  1000, numberEpisodics, 12, -1)
      
      figAlpha2, axAlpha2 = plt.subplots(figsize=(10,4))
      axAlpha2.plot(rewardsMean, label='Promedio evolución recompensa', color = '#F63366')
      axAlpha2.plot([max(banditMeans)+ np.cos(w0*t) for t in range(1000)], label=f'$\mu$', color = 'blue')
      axAlpha2.set_title(f'Evolución de la recompensa promedio en {numberEpisodics} episodios', fontsize = 14)
      axAlpha2.set_ylabel('Recompensa promedio', fontsize = 14)
      axAlpha2.set_xlabel('Iteraciones', fontsize = 14)
      axAlpha2.legend()
      axAlpha2.grid()

      figAlphaBar2, axAlphaBar2 = plt.subplots(figsize=(10,4))
      axAlphaBar2.bar(range(12), meanCountActions, color = '#F63366')
      axAlphaBar2.set_xticks(np.arange(12), np.arange(12), fontsize = 14)
      axAlphaBar2.set_title(f'Porcentaje promedio de acciones tomadas en todos los episodios por $\epsilon$, \n mejor acción {betterAction}', 
                fontsize = 14)
      axAlphaBar2.set_xlabel('Acciones', fontsize = 14)
      axAlphaBar2.set_ylabel('% de cantidad de veces escogida', fontsize = 14)
      axAlphaBar2.grid(axis='y')
  
  col_a1, col_b1= st.columns(2)
  with st.container():
    
    with col_a1:
      st.subheader(rf'Resultados con $\alpha$ = {alpha} y $\epsilon$ = {epsilon}')
      if submitted:
        #st.write(f'Evolución de la recompensa promedio en {numberEpisodics} episodios')
        st.pyplot(figAlpha1)

    with col_b1:
      st.subheader(rf'Resultados con $\alpha = 1/t$  y $\epsilon$ = {epsilon}')
      if submitted:
        #st.subheader(f'Evolución de la recompensa promedio en {numberEpisodics} episodios')
        st.pyplot(figAlpha2)
    
  col_bar1, col_bar2 = st.columns(2)
  
  with st.container():
    #st.subheader(fr'Diagrama de barras del porcentaje promedio de acciones tomadas en todos los episodios por $\epsilon$, mejor acción {betterAction}')
    with col_bar1:
      if submitted:
        st.pyplot(figAlphaBar1)
    
    with col_bar2:
      if submitted:
        st.pyplot(figAlphaBar2)