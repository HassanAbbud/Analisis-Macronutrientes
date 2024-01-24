import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm


prueba = plt.figure(figsize=(20,20))
normalidad = prueba.add_subplot(2,2,1)
varianza =  prueba.add_subplot(2,2,2)
independencia = prueba.add_subplot(2,2,3)
histograma = prueba.add_subplot(2,2,4)


df = pd.read_csv('Datos_Alimentos.csv')

x = df[["Grasas","Proteínas","Carbohidratos"]]#,"Sodio"]]

y = df["Calorías"] 


#x es variable independiente
#y es variable independiente

#x = sm.add_constant(x)
#^^^^opcion para ver como queda la R con la constante ^^^^.

model = sm.OLS(y,x).fit()
print(model.summary())
#La primera vez salio p value de sodio = 0.39, asi que no era significativo

pronosticos = model.predict(x)
residuos = y-pronosticos
residuosordenados = sorted(residuos)

Z = []
for i in range(1, len(residuos)+1):
  Z.append(stats.norm.ppf((i-0.5)/len(residuos)))


normalidad.scatter(residuosordenados,Z)
previo = np.polyfit(residuosordenados,Z,1)
tendencia = np.poly1d(previo)
normalidad.plot(residuosordenados,tendencia(residuosordenados),"r-")
normalidad.set_title('Normalidad')
normalidad.grid(True)

varianza.set_title('Varianza')
varianza.scatter(pronosticos,residuos)
varianza.grid(True)

independencia.set_title('Independencia')
independencia.plot(residuos)
independencia.grid(True)

histograma.set_title('Histograma')
histograma.hist(residuos)
histograma.grid(True)

prueba.savefig('Tablero de pruebas')
prueba.show()