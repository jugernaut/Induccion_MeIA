import numpy as np


# Codificación de las relaciones, primero en un arreglo de cadenas
# Esposas
esposas = np.array([
        ('Christopher', 'tiene-esposa', 'Penelope'),
        ('Andrew', 'tiene-esposa', 'Christine'),
        ('Arthur', 'tiene-esposa', 'Margaret'),
        ('James', 'tiene-esposa', 'Victoria'),
        ('Charles', 'tiene-esposa', 'Jennifer'),
        ('Roberto', 'tiene-esposa', 'Maria'),
        ('Pierro', 'tiene-esposa', 'Francesca'),
        ('Emilio', 'tiene-esposa', 'Gina'),
        ('Marco', 'tiene-esposa', 'Lucia'),
        ('Tomaso', 'tiene-esposa', 'Angela')
        ])
# Esposos
esposos = np.c_[esposas[:,2], np.repeat('tiene-esposo', len(esposas)), esposas[:,0]]

# Padres
padres = np.array([
        ('Arthur', 'tiene-padre', 'Christopher'),
        ('Victoria', 'tiene-padre', 'Christopher'),
        ('James', 'tiene-padre', 'Andrew'),
        ('Jennifer', 'tiene-padre', 'Andrew'),
        ('Colin', 'tiene-padre', 'James'),
        ('Charlotte', 'tiene-padre', 'James'),
        ('Emilio', 'tiene-padre', 'Roberto'),
        ('Gina', 'tiene-padre', 'Roberto'),
        ('Marco', 'tiene-padre', 'Pierro'),
        ('Angela', 'tiene-padre', 'Pierro'),
        ('Alfonso', 'tiene-padre', 'Marco'),
        ('Sophia', 'tiene-padre', 'Marco')
        ])
# Madres
madres = np.c_[padres[:,0], np.repeat('tiene-madre', len(padres)), padres[:,2]]
for i in range(len(padres)):
    madres[i][2] = esposas[esposas[:,0] == madres[i][2]][0,2]

# Hermanos
hermanos = np.array([
        ('Victoria','tiene-hermano','Arthur'),
        ('Jennifer','tiene-hermano','James'),
        ('Charlotte','tiene-hermano','Colin'),
        ('Lucia','tiene-hermano','Emilio'),
        ('Angela','tiene-hermano','Marco'),
        ('Sophia','tiene-hermano','Alfonso')
        ])
hermanas = np.c_[hermanos[:,2], np.repeat('tiene-hermana', len(hermanos)), hermanos[:,0]]

# Hijos
temp = padres[::2]
hijos = np.c_[temp[:,2], np.repeat('tiene-hijo', len(temp)), temp[:,0]]
temp = madres[::2]
hijos = np.vstack((hijos, np.c_[temp[:,2], np.repeat('tiene-hijo', len(temp)), temp[:,0]]))

# Hijas
temp = padres[1::2]
hijas = np.c_[temp[:,2], np.repeat('tiene-hija', len(temp)), temp[:,0]]
temp = madres[1::2]
hijas = np.vstack((hijas, np.c_[temp[:,2], np.repeat('tiene-hija', len(temp)), temp[:,0]]))

# Tíos
tios = np.array([
        ('Colin', 'tiene-tio', 'Arthur'),
        ('Charlotte', 'tiene-tio', 'Arthur'),
        ('Colin', 'tiene-tio', 'Charles'),
        ('Charlotte', 'tiene-tio', 'Charles'),
        ('Alfonso', 'tiene-tio', 'Emilio'),
        ('Sophia', 'tiene-tio', 'Emilio'),
        ('Alfonso', 'tiene-tio', 'Tomaso'),
        ('Sophia', 'tiene-tio', 'Tomaso'),
    ])
# Tías
tias = np.c_[tios[:,0], np.repeat('tiene-tia', len(tios)), tios[:,2]]
for i in range(len(tios)):
    tias[i][2] = esposas[esposas[:,0] == tias[i][2]][0,2]

# Sobrinos
temp = tios[::2]
sobrinos = np.c_[temp[:,2], np.repeat('tiene-sobrino', len(temp)), temp[:,0]]
temp = tias[::2]
sobrinos = np.vstack((sobrinos, np.c_[temp[:,2], np.repeat('tiene-sobrino', len(temp)), temp[:,0]]))

# Sobrinas
temp = tios[1::2]
sobrinas = np.c_[temp[:,2], np.repeat('tiene-sobrina', len(temp)), temp[:,0]]
temp = tias[1::2]
sobrinas = np.vstack((sobrinas, np.c_[temp[:,2], np.repeat('tiene-sobrina', len(temp)), temp[:,0]]))

tercias = np.vstack((esposas, esposos, padres, madres, hermanos, hermanas,
                        hijos, hijas, tios, tias, sobrinos, sobrinas))


# De todas las relaciones, eliminaremos la persona 2.
# Las entradas distintas que quedan son:
unicas = np.vstack(list({tuple(row) for row in tercias[:,0:2]}))


# Pasar las relaciones a vectores de entrada numéricos para la red neuronal
# Tómese en cuenta que, para cada entrada, puede haber más de una respuesta.

total = len(unicas)
Datos = np.zeros((total, 36))
Salidas = np.zeros((total, 24))

personas = ['Christopher', 'Penelope', 'Andrew', 'Christine',
            'Margaret', 'Arthur', 'Victoria', 'James', 'Jennifer', 'Charles',
            'Colin', 'Charlotte',
            'Roberto', 'Maria', 'Pierro', 'Francesca',
            'Gina', 'Emilio', 'Lucia', 'Marco', 'Angela', 'Tomaso',
            'Alfonso', 'Sophia'
           ]
relaciones = ['tiene-'+r for r in ['esposa', 'esposo', 'padre', 'madre', 'hermano', 'hermana',
                        'hijo', 'hija', 'tio', 'tia', 'sobrino', 'sobrina']]

indiceNeuronaPersona = {}
for i, persona in enumerate(personas):
    indiceNeuronaPersona[persona] = i
indiceNeuronaRelacion = {}
for i, rel in enumerate(relaciones):
    indiceNeuronaRelacion[rel] = i

for i, renglon in enumerate(unicas):
    renglonesCorrespondientes = tercias[np.all(tercias[:,0:2] == renglon, axis=1)]
    ##print(i, renglon, renglonesCorrespondientes)
    # Persona
    Datos[i][indiceNeuronaPersona[renglon[0]]] = 1.0
    # Relación
    Datos[i][len(personas) + indiceNeuronaRelacion[renglon[1]]] = 1.0
    ##print("Entrada: ", Datos[i])
    # Respuesta(s)
    for respuesta in renglonesCorrespondientes:
        ##print(respuesta)
        Salidas[i][indiceNeuronaPersona[respuesta[2]]] = 1.0
    ##print("Salida: ", Salidas[i])
