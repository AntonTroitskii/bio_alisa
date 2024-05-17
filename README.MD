Ссылка на задание - https://contest.yandex.ru/contest/28413/problems/F/

# F. Биометрия

Ограничение времени 2 секунды

Ограничение памяти  122Mb

Ввод    стандартный

Вывод   стандартный вывод или answers.tsv

В данной задаче нужно по звуковым файлам в формате wav предсказать пол человека, речь которого записана на каждом из файлов (0 – мужчина, 1 – женщина).

Для того, чтобы получить OK по этой задаче, нужно получить точность более 98 процентов на тестовом наборе данных.

Тренировочный набор данных: https://disk.yandex.ru/d/IUUTPJFOfwn_OQ. В тренировочном наборе данных есть файл targets.tsv, в котором находятся правильные значения пола для всех записей в тренировочном наборе данных.

Тестовый набор данных: https://yadi.sk/d/K8Z-_gQbspmxkhw. В систему нужно отправить файл аналогичный targets.tsv из тренировочного набора. То есть для каждого файла id.wav в тестовом наборе данных в файле ответа должна быть строка вида

id\tgender

Примечания

Обратите внимание, что в данной задаче нужно прислать не код, а файл с результатом.

Для того, чтобы из звуковых файлов получить признаки, можно воспользоваться функцией https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html, которая для звукового файла вычислит его спектрограмму, то есть матрицу размера число признаков (задаётся параметром n_mels) на длину записи.

Если получен WA, чтобы узнать причину, можно посмотреть отчёт по посылке и увидеть его в выводе постпроцессора. Причиной такого вердикта может быть как неправильный формат вывода, так и недостаточная точность.

# Pipeline
- преобхоавание *.wav файлов в мелспекторкграммы формата .png (используется паралельное обработка файлов из архива);
- transfer learning для обучения на спектограммах с помощью pytorch;
- для контоля состояния dataset использовался DVC.

# Как можно улучишть проект
- можно выполшнять обаботку .wav файлов без предварительного сохранения данных в виде .png. Как описано тут: Audio Deep Learning Made Simple: Sound Classification,
        Step-by-Step - [towardsdatascience.com](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)

# Полезное

-   Как решать подобные задачи?

    -   Audio Deep Learning Made Simple (Part 1): State-of-the-Art
        Techniques -
        [towardsdatascience.com](https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504)

    -   Audio Deep Learning Made Simple (Part 2): Why Mel Spectrograms
        perform better -
        [towardsdatascience.com](https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505)

    -   Audio Deep Learning Made Simple: Sound Classification,
        Step-by-Step - [towardsdatascience.com](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)

    -   <https://stackoverflow.com/questions/55513652/which-spectrogram-best-represents-features-of-an-audio-file-for-cnn-based-model>

    -   Train Test Split in Deep Learning -
        [www.lightly.ai](https://www.lightly.ai/post/train-test-split-in-deep-learning)

    -   ![](media/image1.png){width="0.13333333333333333in"
        height="0.13333333333333333in"} Введение в аудиоданные -
        [huggingface.co](https://huggingface.co/learn/audio-course/ru/chapter1/audio_data)
        Очень круто рассказано про мелспектрограммы.

-   Librosa

    -   Read zip -
        [librosa.org](https://librosa.org/doc/0.10.1/ioformats.html#read-file-like-objects)

    -   librosa.power_to_db -
        [librosa.org](https://librosa.org/doc/main/generated/librosa.power_to_db.html)

-   Unzip архив целиком, потом обработать.

    -   python -m zipfile \--extract sample.zip - Разархивировать zip
        файл из командной строки.

-   PILLOW

    -   Введение в обработку изображений в Python при помощи Pillow-
        [habr.com](https://habr.com/ru/articles/530800/)

-   Чтение данных байтового

    -   <https://stackoverflow.com/questions/74023061/loading-a-compressed-file-with-librosa>

    -   <https://stackoverflow.com/questions/59426275/download-and-open-file-with-librosa-without-writing-to-filesystem>

-   Zip parallel processing

    -   <https://stackoverflow.com/questions/54736710/multiprocessing-pool-maybeencodingerror-typeerrorcannot-serialize-io-buffe>

    -   <https://stackoverflow.com/questions/54507527/how-can-i-fix-typeerror-cannot-serialize-io-bufferedreader-object-error-wh>

-   Python

    -   Multiprocessing

        -   <https://stackoverflow.com/questions/54507527/how-can-i-fix-typeerror-cannot-serialize-io-bufferedreader-object-error-wh>

        -   <https://superfastpython.com/multithreaded-unzip-files/>

        -   ![](media/image1.png){width="0.13333333333333333in"
            height="0.13333333333333333in"} Python для сетевых
            инженеров -
            [pyneng.readthedocs.io](https://pyneng.readthedocs.io/ru/latest/index.html)

        -   Зачем, когда и как использовать multithreading и
            multiprocessing в Python -
            [habr.com](https://habr.com/ru/companies/otus/articles/501056/)

        -   Performance gradually decreases while running Python
            multiprocess -
            [stackoverflow.com](https://stackoverflow.com/questions/61198025/performance-gradually-decreases-while-running-python-multiprocess)

        -   How to use multiprocessing pool.map with multiple
            arguments -
            [stackoverflow.com](https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments)

    -   Batches

        -   Split a python list into other \"sublists\" i.e smaller
            lists \[duplicate\] -
            [stackoverflow.com](https://stackoverflow.com/questions/9671224/split-a-python-list-into-other-sublists-i-e-smaller-lists)

    -   Pathlib

        -   How to Get a List of All Files in a Directory With Python -
            [realpython.com](https://realpython.com/get-all-files-in-directory-python/)

    -   Itertools

        -   Python -- Itertools.islice() -
            [www.geeksforgeeks.org](https://www.geeksforgeeks.org/python-itertools-islice/)

    -   Loging

        -   Логирование в Python: руководство разработчика -
            [habr.com](https://habr.com/ru/companies/wunderfund/articles/683880/)

        -   Модуль mu -
            [pyneng.readthedocs.io](https://pyneng.readthedocs.io/ru/latest/book/19_concurrent_connections/logging.html)

    -   Pickle

        -   Save and Load -
            [www.geeksforgeeks.org](https://www.geeksforgeeks.org/how-to-use-pickle-to-save-and-load-variables-in-python/)

-   Pipenv. Pip

    -   Package location -
        [stackoverflow.com](https://stackoverflow.com/questions/122327/how-do-i-find-the-location-of-my-python-site-packages-directory)

        -   pip show pipenv

    -   pipenv install. fbgemm.dll -
        [stackoverflow.com](https://stackoverflow.com/questions/78114412/import-torch-how-to-fix-oserror-winerror-126-error-loading-fbgemm-dll-or-depen)

        -   reinstall pyenv

-   Kaggle

    - [No module named
    \'\_ctypes\'](https://stackoverflow.com/questions/63996623/no-module-named-ctypes) -
    [stackoverflow.com](https://stackoverflow.com/questions/63996623/no-module-named-ctypes)

-   Fast.AI

    -   Training Imagenette -
        [docs.fast.ai](https://docs.fast.ai/tutorial.imagenette.html)

    -   Vision data -
        [docs.fast.ai](https://docs.fast.ai/vision.data.html)

    -   Get Result for Test

        -   How to get prediction using get_preds in fastai 2? -
            [fкorums.fast.ai](https://forums.fast.ai/t/how-to-get-prediction-using-get-preds-in-fastai-2/85597)

    -   How to know if fastai is using the NVIDIA GPU -
        [forums.fast.ai](https://forums.fast.ai/t/how-to-know-if-fastai-is-using-the-nvidia-gpu/45605/13)

    -   Device

        -   <https://docs.fast.ai/vision.data.html#imagedataloaders.from_df>

-   GPU computations

    -   Ваша первая нейронная сеть на графическом процессоре (GPU).
        Руководство для начинающих -
        <https://habr.com/ru/articles/488564/>

    -   Как GPU-вычисления буквально спасли меня на работе. Пример на
        Python -
        [habr.com](https://habr.com/ru/companies/piter/articles/505590/)

    -   fast.ai not using the GPU -
        [stackoverflow.com](https://stackoverflow.com/questions/70351366/fast-ai-not-using-the-gpu)

    -   cuda

        -   pipenv install cuda-python

Torch

-   Torch install

    - <https://github.com/pypa/pipenv/issues/4961#issuecomment-1045679643>

    -   <https://www.nvidia.com/download/driverResults.aspx/143680/en-us/>

    -   <https://medium.com/analytics-vidhya/installing-any-version-of-cuda-on-ubuntu-and-using-tensorflow-and-torch-on-gpu-b1a954500786>

<!-- -->

-   Torch. LoadData

    -   <https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/>

-   Torch

    -   Subset -
        [stackoverflow.com](https://stackoverflow.com/questions/54251798/pytorch-can-not-slice-torchvision-mnist-dataset)

<!-- -->

-   ResNet

    -   How can I use a pre-trained neural network with grayscale
        images? -
        [stackoverflow.com](https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images)

-   DVC

    -   DVC shared

        -   <https://drive.google.com/drive/folders/1XYYwhoXprp3ro-DMcrFHuk5-rNaY3f6x?usp=sharing>

            -   1XYYwhoXprp3ro-DMcrFHuk5-rNaY3f6x

        -   ERROR: unexpected error - Failed to authenticate GDrive:
            Access token refresh failed: invalid_grant: Bad Request

            -   <https://discuss.dvc.org/t/manually-prompt-gdrive-authentication-step/858>

            -   Помогло. Поменял файл для хранения этого

                > dvc remote modify myremote \--local \\
                >
                >       gdrive_user_credentials_file
                > \~/.gdrive/myremote-credentials.json

-   DVC Large Dataset

    -   Did you know DVC doesn\'t handle large datasets? Neither did we
        and here\'s how we fixed it-
        [fizzylogic.nl](https://fizzylogic.nl/2023/01/13/did-you-know-dvc-doesn-t-handle-large-datasets-neither-did-we-and-here-s-how-we-fixed-it)

    -   Guide to File Formats for Machine Learning: Columnar, Training,
        Inferencing, and the Feature Store -
        [towardsdatascience.com](https://towardsdatascience.com/guide-to-file-formats-for-machine-learning-columnar-training-inferencing-and-the-feature-store-2e0c3d18d4f9)

    -   [Loading .npy files as dataset for
        pytorch](https://stackoverflow.com/questions/57989716/loading-npy-files-as-dataset-for-pytorch)

    -   Why I Chose WebDataset for Training on 50TB of Data?-
        [medium.com](https://medium.com/red-buffer/why-did-i-choose-webdataset-for-training-on-50tb-of-data-98a563a916bf)

    -   <https://webdataset.github.io/webdataset/creating/#using-tar>

    -   WebDataset

        -   Getting Started -
            [webdataset.github.io](https://webdataset.github.io/webdataset/gettingstarted/)

<!-- -->

-   Matplotlib. Savefig

    -   <https://www.tutorialspoint.com/how-to-convert-matplotlib-figure-to-pil-image-object>

    -   3 Methods to Save Plots as Images or PDF Files in Matplotlib-
        [medium.com](https://medium.com/@akaivdo/3-methods-to-save-plots-as-images-or-pdf-files-in-matplotlib-96a922fd2ce4)

    -   Using memory

-   Download file

    -   Download large file in Python with beautiful progress bar -
        [pub.aimind.so](https://pub.aimind.so/download-large-file-in-python-with-beautiful-progress-bar-f4f86b394ad7)

    -   Progress Bar while download file over http with Requests -
        [stackoverflow.co](https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests)

    -   \-
        <https://www.alpharithms.com/progress-bars-for-python-downloads-580122/>

    -   yandes file download  -
        [ru.stackoverflow.com](https://ru.stackoverflow.com/questions/1088300/как-скачивать-файлы-с-яндекс-диска)

    -   tqdm has no attribute wrapattr -
        [cainmagi.github.io](https://cainmagi.github.io/FFmpeg-Encoder-Decoder-for-Python/docs/next/troubleshooting/installation#tqdm-has-no-attribute-wrapattr)

    -   [Easy Python Progress Bars with
        tqdm](https://www.youtube.com/watch?v=n4E7of9BINo)

-   GIT

    -   [Git - List all files currently under source
        control?](https://superuser.com/questions/429693/git-list-all-files-currently-under-source-control