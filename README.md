# Отчёт

Стандартный [pipline](https://github.com/EgorOs/osinkin_hw10/blob/master/tutorial/from_class.py#L46) из туториала был дополнен [токенайзером](https://github.com/EgorOs/osinkin_hw10/blob/master/with_lemma.py#L17), который преобразует слова в леммы, а также удаляет заголовки файлов и символы, не являющиеся знаками препинания. Результаты работы классификаторов приведены ниже:

##### Результат работы исходного классификатора.
![Test 1](https://raw.githubusercontent.com/EgorOs/osinkin_hw10/master/pics/no_lemma.png)
##### Данные лемматизированы, заголовки файлов удалены.
![Test 2](https://raw.githubusercontent.com/EgorOs/osinkin_hw10/master/pics/lemma_no_header.png)
##### Данные лемматизированы, заголовки файлов сохранены.
![Test 3](https://raw.githubusercontent.com/EgorOs/osinkin_hw10/master/pics/lemma_header.png)

### Вывод:
Как видно из результатов тестов, наличие заголовков файлов, содержащий информацию о сообщении, влияет на точность результата.
