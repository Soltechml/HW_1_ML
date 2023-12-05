# HW_1_ML

**Что было сделано?**

*Подготовка и обработка данных*

1. Считали 2 csv-файла, в train и в test
2. Отобразили некоторые строки дата-сета, визуально оценив признаки
3. Нашли и удалили повторяющиеся строки
4. Нашли пропуски в признаках и заполнили их медианами

   
*Провели визуализация данных*

5. Построили попарные распределения всех числовых признаков для трейна и для теста
6. Построили тепловую карту попарных корреляций числовых колонок для трейна
7. Отобразили диаграмму рассеяния для наиболее скореллированной пары признаков
8. Произвели дополнительные визуализации, построив гистограмму распределения цен на автомобили и отобразили линейную зависимость между максимальной мощностью и ценой продажи.


*Построили линейные модели*

9. Модель на вещественных признаках: year	km_driven	mileage	engine	max_power	seats	torque_value	max_torque_rpm. Получили ошибку R^2 = 0.59 на трейне и на тесте.
10. Следующим этапом произвели стандартизацию вещественных признаков и пересчитали модель на стандартизированных признаках. Получили аналогичную R^2 = 0.59.
11. Определили самый информативный признак, которым оказался: max_power
12. Обучили Lasso-регрессию, продолжив обучать модели на нормализованных признаках
13. Перебором по сетке (c 10-ю фолдами) подобрали оптимальные параметры для Lasso-регрессии
14. Определили лучшие гипер-параметры

15. Добавили категориальные фичи, сделали кодирование методом OneHot
16. Посчитали гребневую регрессию, улучшив значение  R^2 до 0.63 на тестовых данных
  

*Реализовали бизнесовую часть*

17. Посчитали кастомную метрику -- среди всех предсказанных цен на авто долю предиктов, отличающихся от реальных цен на эти авто не более чем на 10%


*Попытались реализовать продуктовую часть*

18. Много времени потратил на интеграцию api, но в итоге оно так и не завелось, обидно, даже дедлайн просрочил, в надежде сделать интеграцию. Сервис на FastApi должен принимать json c признаками одного объекта или csv-файл с несколькими объектами, и на выходе выдавать прогноз по цене.
