import os                # Библиотека для работы с операционной системой
import numpy as np       # для работы с математическими функциями
import cv2               # для обработки лиц
import face_recognition  # для определения лиц

database = 'Database'            # Путь к папке с фотографиями-образцами
photoList = []                   # Массив для фотографий
namesList = []                   # Массив для имен людей
dataList = os.listdir(database)  # Список файлов в папке

for i in dataList:               # ввод данных в массивы
    pathImg = cv2.imread(f'{database}/{i}')
    photoList.append(pathImg)
    namesList.append(os.path.splitext(i)[0])
print('Зарегистрированные пользователи: ', namesList)  # вывод имен зарегистрированных людей


def Encoding(images):   # кодирование фотографий в векторы
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = Encoding(photoList)  # массив векторов зарегистрированных людей
# Ошибка возникает в случае, когда на фотографии-эталоне (хранящейся в папке с образцами) не обнаружено лицо

webcamera = cv2.VideoCapture(0)

while True:     # покадровая бработка видеопотока с вэб-камеры
    success, img = webcamera.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    findFaces = face_recognition.face_locations(imgS)                # поиск всех лиц в текущем кадре
    encodeFaces = face_recognition.face_encodings(imgS, findFaces)   # кодирвание найденных лиц в векторы


    # Цикл для распознавания лиц
    for encodeFace, faceLocation in zip(encodeFaces, findFaces):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)   # возвращает True/False при сравнении лица с эталонами из базы данных
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)   # возвращает евклидово пространство при сравнении эталонами из базы данных
        matchIndex = np.argmin(faceDis)  # возвращает индекс минимального евклидова пространства

        name = 'Unknown'
        # Изначально имя для найденного лица будет неизвестно, при распознавании оно заменится, иначе - оно выведется
        if matches[matchIndex]:   # Поиск, выделение и подпись лица на видео
            name = namesList[matchIndex]
            result_text = "{0} {1}".format(name,round(min(faceDis),3))
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(img, (x1-1, y2 + 35), (x2+1, y2), (0, 255, 255), cv2.FILLED)
            cv2.putText(img, result_text, (x1 + 6, y2 + 27), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

        else:  # Если человек не определен, то просто выводится надпись "Unknown"
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1-1, y2 + 35), (x2+1, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 + 27), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Scanner", img)  # вывод видеопотока на экран
    cv2.waitKey(1)   # 0 – следующий кадр по нажатию кнопки
                     # 1 – постоянный поток с камеры

