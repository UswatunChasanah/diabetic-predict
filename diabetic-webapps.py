from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
from scipy.io.arff import loadarff
from io import StringIO, BytesIO
import urllib.request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Dataset", "Pre-processing",
                 "Modeling", "Implementasi"],
        icons=["house", "clipboard-data", "wrench",
               "diagram-3", "pc-display-horizontal"],
        menu_icon="cast",
        default_index=0,
    )

df = pd.read_csv(
    "https://raw.githubusercontent.com/UswatunChasanah/datamining/gh-pages/diabetes_data_upload.csv")
if selected == "Home":
    st.write("""
    # Web Apps - Deteksi Diabetes
    Aplikasi Berbasis Web untuk Deteksi Penyakit Diabetes
    ### Indonesia Sehat
    ## Cegah Diabetes Sejak Dini
    Diabetes atau penyakit gula adalah penyakit kronis atau yang berlangsung jangka panjang. Penyakit ini ditandai dengan meningkatnya kadar gula darah (glukosa) hingga di atas nilai normal. Diabetes terjadi ketika tubuh pengidapnya tidak lagi mampu mengambil gula (glukosa) ke dalam sel dan menggunakannya sebagai energi. Kondisi ini pada akhirnya menghasilkan penumpukan gula ekstra dalam aliran darah tubuh.
    """
             )
    image = Image.open('home.jpg')
    st.image(image)
if selected == "Dataset":
    st.markdown("<h1 style='text-align: center;'>Data Set</h1>",
                unsafe_allow_html=True)
    st.write("""
    ## Penjelasan Dataset
    Dataset ini adalah kumpulan data prediksi resiko diabetes tahap awal yang berisi data tanda dan gejala penderita diabetes baru atau yang beresiko menjadi penderita diabetes. Data ini dikumpulkan menggunakan kuisioner langsung dari pasien rumah sakit diabetes Sylhet di Sylhet, Bangladesh.
    ## Deskripsi Data
    Di dalam dataset ini terdiri dari 17 fitur dengan deskripsi dibawah ini :
    1. Age : Digunakan untuk mengetahui usia dari pasien. Type data dari fitur age adalah numerik, contoh nilai age : 30, 40, 50.
    2. Gender : Digunakan untuk mengetahui jenis kelamin pasien. Type data dari fitur gender adalah kategorikal berupa male dan female.
    3. Polyuria : Digunakan untuk mengetahui apakah pasien sering buang air kecil atau tidak. Type data dari fitur polyuria adalah kategorikal berupa Yes dan No.
    4. Polydipsi : Digunakan untuk mengetahui apakah pasien merasa haus yang berlebihan atau tidak. Type data dari fitur polydipsi adalah kategorikal berupa Yes dan No.
    5. Sudden weight loss : Digunakan untuk mengetahui apakah pasien mengalami penurunan berat badan secara tiba-tiba atau tidak. Type data dari fitur sudden weight loss adalah kategorikal berupa Yes dan No.
    6. Weakness : Digunakan untuk mengetahui apakah pasien mengalami kelemahan di salah satu sisi tubuh atau tidak. Type data dari fitur weakness adalah kategorikal berupa Yes dan No.
    7. Polyphagia : Digunakan untuk mengetahui apakah pasien mengalami meningkatnya nafsu makan atau rasa lapar berlebih atau tidak. Type data dari fitur polyphagia adalah kategorikal berupa Yes dan No.
    8. Genital thrush : Digunakan untuk mengetahui apakah pasien mengalami infeksi kelamin atau tidak. Type data dari fitur genital thrush adalah kategorikal berupa Yes dan No.
    9. Visual blurring : Digunakan untuk mengetahui apakah pasien mengalami pandangan kabur atau tidak. Type data dari fitur visual blurring adalah kategorikal berupa Yes dan No.
    10. Itching : Digunakan untuk mengetahui apakah pasien mengalami gatal atau tidak. Type data dari fitur itching adalah kategorikal berupa Yes dan No.
    11. Irritability : Digunakan untuk mengetahui apakah pasien mengalami iritasi atau tidak. Type data dari fitur irritability adalah kategorikal berupa Yes dan No.
    12. Delayed healing : Digunakan untuk mengetahui apakah pasien mengalami penyambuhan luka yang berlangsung lama atau tidak. Type data dari fitur delayed healing adalah kategorikal berupa Yes dan No.
    13. Partial paresis : Digunakan untuk mengetahui apakah pasien mengalami gangguan gerakan pada sebagian tubuh atau tidak. Type data dari fitur partial paresis adalah kategorikal berupa Yes dan No.
    14. Muscle stiffness : Digunakan untuk mengetahui apakah pasien mengalami otot kaku atau tidak. Type data dari fitur muscle stiffness adalah kategorikal berupa Yes dan No.
    15. Alopecia : Digunakan untuk mengetahui apakah pasien mengalami kerontokan rambut atau tidak. Type data dari fitur alopecia adalah kategorikal berupa Yes dan No.
    16. Obesity : Digunakan untuk mengetahui apakah pasien mengalami obesitas atau tidak. Type data dari fitur obesity adalah kategorikal berupa Yes dan No.
    17. Class : Class merupakan hasil dari klasifikasi semua fitur yang ada apakah pasien mengidap penyakit diabetes mellitus atau bisa disebut dengan label. Type data dari fitur class adalah kategorikal berupa positive dan negative.
    """)

    st.markdown("<h3 style='text-align: center;'>Dataset Diabetes</h3>",
                unsafe_allow_html=True)
    st.dataframe(df)

if selected == "Pre-processing":
    st.markdown("<h1 style='text-align: center;'>Pre-Processing</h1>",
                unsafe_allow_html=True)
    st.write("""
    Data Preprocessing merupakan salah satu tahapan dalam melakukan mining data. Sebelum menuju ke tahap  pemprosesan. Data mentah akan diolah terlebih dahulu. Data Preprocessing atau praproses data biasanya dilakukan melalui cara eliminasi data yang tidak sesuai. Selain itu dalam proses ini data akan diubah dalam bentuk yang akan lebih dipahami oleh sistem.
    Pengertian lain menyebutkan bahwa data preprocessing adalah tahapan untuk menghilangkan beberapa permasalahan yang bisa mengganggu saat pemrosesan data. Hal tersebut karena banyak data yang formatnya tidak konsisten. Data preprocessing merupakan teknik paling awal sebelum melakukan data mining. Namun terdapat beberapa proses juga dalam data preprocessing seperti membersihkan, mengintegrasikan, mentransformasikan dan mereduksi data.
    ## 1. Encoder Data
    Encoder data dilakukan untuk merubah tipe data dari kategorikal menjadi numerik.
    #### Dataset sebelum di encoder
    """)
    st.dataframe(df)

    st.write("""
    #### Dataset setelah di encoder
    """)

    df['class'] = pd.Categorical(df["class"])
    df["class"] = df["class"].cat.codes

    df['Polyuria'] = pd.Categorical(df["Polyuria"])
    df["Polyuria"] = df["Polyuria"].cat.codes

    df['Polydipsia'] = pd.Categorical(df["Polydipsia"])
    df["Polydipsia"] = df["Polydipsia"].cat.codes

    df['Gender'] = pd.Categorical(df["Gender"])
    df["Gender"] = df["Gender"].cat.codes

    df['sudden weight loss'] = pd.Categorical(df["sudden weight loss"])
    df["sudden weight loss"] = df["sudden weight loss"].cat.codes

    df['weakness'] = pd.Categorical(df["weakness"])
    df["weakness"] = df["weakness"].cat.codes

    df['Polyphagia'] = pd.Categorical(df["Polyphagia"])
    df["Polyphagia"] = df["Polyphagia"].cat.codes

    df['Genital thrush'] = pd.Categorical(df["Genital thrush"])
    df["Genital thrush"] = df["Genital thrush"].cat.codes

    df['visual blurring'] = pd.Categorical(df["visual blurring"])
    df["visual blurring"] = df["visual blurring"].cat.codes

    df['Itching'] = pd.Categorical(df["Itching"])
    df["Itching"] = df["Itching"].cat.codes

    df['Irritability'] = pd.Categorical(df["Irritability"])
    df["Irritability"] = df["Irritability"].cat.codes

    df['delayed healing'] = pd.Categorical(df["delayed healing"])
    df["delayed healing"] = df["delayed healing"].cat.codes

    df['partial paresis'] = pd.Categorical(df["partial paresis"])
    df["partial paresis"] = df["partial paresis"].cat.codes

    df['muscle stiffness'] = pd.Categorical(df["muscle stiffness"])
    df["muscle stiffness"] = df["muscle stiffness"].cat.codes

    df['Alopecia'] = pd.Categorical(df["Alopecia"])
    df["Alopecia"] = df["Alopecia"].cat.codes

    df['Obesity'] = pd.Categorical(df["Obesity"])
    df["Obesity"] = df["Obesity"].cat.codes

    st.dataframe(df)

if selected == "Modeling":
    df['class'] = pd.Categorical(df["class"])
    df["class"] = df["class"].cat.codes

    df['Polyuria'] = pd.Categorical(df["Polyuria"])
    df["Polyuria"] = df["Polyuria"].cat.codes

    df['Polydipsia'] = pd.Categorical(df["Polydipsia"])
    df["Polydipsia"] = df["Polydipsia"].cat.codes

    df['Gender'] = pd.Categorical(df["Gender"])
    df["Gender"] = df["Gender"].cat.codes

    df['sudden weight loss'] = pd.Categorical(df["sudden weight loss"])
    df["sudden weight loss"] = df["sudden weight loss"].cat.codes

    df['weakness'] = pd.Categorical(df["weakness"])
    df["weakness"] = df["weakness"].cat.codes

    df['Polyphagia'] = pd.Categorical(df["Polyphagia"])
    df["Polyphagia"] = df["Polyphagia"].cat.codes

    df['Genital thrush'] = pd.Categorical(df["Genital thrush"])
    df["Genital thrush"] = df["Genital thrush"].cat.codes

    df['visual blurring'] = pd.Categorical(df["visual blurring"])
    df["visual blurring"] = df["visual blurring"].cat.codes

    df['Itching'] = pd.Categorical(df["Itching"])
    df["Itching"] = df["Itching"].cat.codes

    df['Irritability'] = pd.Categorical(df["Irritability"])
    df["Irritability"] = df["Irritability"].cat.codes

    df['delayed healing'] = pd.Categorical(df["delayed healing"])
    df["delayed healing"] = df["delayed healing"].cat.codes

    df['partial paresis'] = pd.Categorical(df["partial paresis"])
    df["partial paresis"] = df["partial paresis"].cat.codes

    df['muscle stiffness'] = pd.Categorical(df["muscle stiffness"])
    df["muscle stiffness"] = df["muscle stiffness"].cat.codes

    df['Alopecia'] = pd.Categorical(df["Alopecia"])
    df["Alopecia"] = df["Alopecia"].cat.codes

    df['Obesity'] = pd.Categorical(df["Obesity"])
    df["Obesity"] = df["Obesity"].cat.codes

    X = df.drop(columns=['class'])

    cls = df['class'].values

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    cls_encoder = le.fit_transform(cls)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, cls_encoder, test_size=0.3, random_state=1)

    st.markdown("<h1 style='text-align: center;'>Modeling</h1>",
                unsafe_allow_html=True)
    st.write("""
    ### 1. Naive Bayes
    Naive Bayes adalah algoritma machine learning yang digunakan untuk keperluan klasifikasi atau pengelompokan suatu data. Algoritma ini didasarkan pada teorema probabilitas yang dikenalkan oleh ilmuwan Inggris Thomas Bayes. Naive Bayes berfungsi memprediksi probabilitas di masa depan berdasarkan pengalaman sebelumnya, sehingga dapat digunakan untuk pengambilan keputusan.
    Ada tiga jenis model Naive Bayes :

    ###### a. Gaussian Naive Bayes
    Ini adalah pengklasifikasi Naive Bayes paling sederhana yang memiliki asumsi bahwa data dari masing-masing label diambil dari distribusi Gaussian sederhana.

    ###### b. Multinomial Naive Bayes
    Pengklasifikasi Naive Bayes lain yang berguna adalah Multinomial Naive Bayes di mana fitur-fiturnya adalah diasumsikan diambil dari distribusi Multinomial sederhana. Naive Bayes semacam itu adalah paling sesuai untuk fitur yang mewakili jumlah diskrit.

    ###### c. Bernoulli Naive Bayes
    Model penting lainnya adalah Bernoulli Naive Bayes di mana fitur diasumsikan biner (0s dan 1s). Klasifikasi teks dengan model 'bag of words' dapat menjadi aplikasi dari Bernoulli Naive Bayes.

    """)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    akurasi_gnb = gnb.score(X_test, y_test) * 100

    st.write("**_Hasil akurasi : ", round(akurasi_gnb, 2), "%_**")

    st.write("""
    ### 2. KNN
    Algoritma KNN atau K-Nearest Neighbor adalah salah satu algoritma yang banyak digunakan di dunia machine learning untuk kasus klasifikasi. algoritma KNN merupakan algoritma klasifikasi yang bekerja dengan mengambil sejumlah K data terdekat (tetangganya) sebagai acuan untuk menentukan kelas dari data baru. Algoritma ini mengklasifikasikan data berdasarkan similarity atau kemiripan atau kedekatannya terhadap data lainnya.
    """)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    akurasi_knn = knn.score(X_test, y_test) * 100
    st.write("**_Hasil akurasi = ", round(akurasi_knn, 2), "%_**")

    st.write("""
    ### 3. Decision Tree
    Decision tree adalah algoritma machine learning yang menggunakan seperangkat aturan untuk membuat keputusan dengan struktur seperti pohon yang memodelkan kemungkinan hasil, biaya sumber daya, utilitas dan kemungkinan konsekuensi atau resiko. Konsepnya adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat, yang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan. 
    """)

    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    akurasi_dt = dt.score(X_test, y_test) * 100
    st.write("**_Hasil akurasi = ", round(akurasi_dt, 2), "%_**")

    st.write("""
    ### 4. Random Forest
    Random Forest adalah algoritma pembelajaran yang supervised. "Forest" yang dibangunnya adalah kumpulan pohon keputusan, biasanya dilatih dengan metode "bagging". Ide umum dari metode bagging adalah kombinasi model pembelajaran meningkatkan hasil keseluruhan.
    """)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X_train, y_train)

    akurasi_rf = rf.score(X_test, y_test) * 100
    st.write("**_Hasil akurasi = ", round(akurasi_rf, 2), "%_**")

if selected == "Implementasi":

    df['class'] = pd.Categorical(df["class"])
    df["class"] = df["class"].cat.codes

    df['Polyuria'] = pd.Categorical(df["Polyuria"])
    df["Polyuria"] = df["Polyuria"].cat.codes

    df['Polydipsia'] = pd.Categorical(df["Polydipsia"])
    df["Polydipsia"] = df["Polydipsia"].cat.codes

    df['Gender'] = pd.Categorical(df["Gender"])
    df["Gender"] = df["Gender"].cat.codes

    df['sudden weight loss'] = pd.Categorical(df["sudden weight loss"])
    df["sudden weight loss"] = df["sudden weight loss"].cat.codes

    df['weakness'] = pd.Categorical(df["weakness"])
    df["weakness"] = df["weakness"].cat.codes

    df['Polyphagia'] = pd.Categorical(df["Polyphagia"])
    df["Polyphagia"] = df["Polyphagia"].cat.codes

    df['Genital thrush'] = pd.Categorical(df["Genital thrush"])
    df["Genital thrush"] = df["Genital thrush"].cat.codes

    df['visual blurring'] = pd.Categorical(df["visual blurring"])
    df["visual blurring"] = df["visual blurring"].cat.codes

    df['Itching'] = pd.Categorical(df["Itching"])
    df["Itching"] = df["Itching"].cat.codes

    df['Irritability'] = pd.Categorical(df["Irritability"])
    df["Irritability"] = df["Irritability"].cat.codes

    df['delayed healing'] = pd.Categorical(df["delayed healing"])
    df["delayed healing"] = df["delayed healing"].cat.codes

    df['partial paresis'] = pd.Categorical(df["partial paresis"])
    df["partial paresis"] = df["partial paresis"].cat.codes

    df['muscle stiffness'] = pd.Categorical(df["muscle stiffness"])
    df["muscle stiffness"] = df["muscle stiffness"].cat.codes

    df['Alopecia'] = pd.Categorical(df["Alopecia"])
    df["Alopecia"] = df["Alopecia"].cat.codes

    df['Obesity'] = pd.Categorical(df["Obesity"])
    df["Obesity"] = df["Obesity"].cat.codes

    from sklearn.preprocessing import MinMaxScaler
    X = df.drop(columns=['class'])

    cls = df['class'].values

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    cls_encoder = le.fit_transform(cls)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, cls_encoder, test_size=0.3, random_state=1)

    st.header("Parameter Inputan")
    umur = st.number_input("Berapa umur anda ?")

    gender = st.radio("Apa gender anda ?", ('Male', 'Female'))
    if gender == 'Male':
        gender = 1
    elif gender == 'Female':
        gender = 0

    polyuria = st.radio("Apakah anda sering buang air kecil ?", ('Yes', 'No'))
    if polyuria == 'Yes':
        polyuria = 1
    elif polyuria == 'No':
        polyuria = 0

    polydipsia = st.radio(
        "Apakah anda merasa haus yang berlebihan ?", ('Yes', 'No'))
    if polydipsia == 'Yes':
        polydipsia = 1
    elif polydipsia == 'No':
        polydipsia = 0

    sudden_weight_loss = st.radio(
        "Apakah anda mengalami penurunan berat badan secara tiba-tiba ?", ('Yes', 'No'))
    if sudden_weight_loss == 'Yes':
        sudden_weight_loss = 1
    elif sudden_weight_loss == 'No':
        sudden_weight_loss = 0

    weakness = st.radio(
        "Apakah anda mengalami kelemahan di salah satu sisi tubuh ?", ('Yes', 'No'))
    if weakness == 'Yes':
        weakness = 1
    elif weakness == 'No':
        weakness = 0

    polyphagia = st.radio(
        "Apakah anda mengalami peningkatan nafsu makan atau rasa lapar berlebihan ?", ('Yes', 'No'))
    if polyphagia == 'Yes':
        polyphagia = 1
    elif polyphagia == 'No':
        polyphagia = 0

    genital_thrush = st.radio(
        "Apakah anda mengalami infeksi kelamin ?", ('Yes', 'No'))
    if genital_thrush == 'Yes':
        genital_thrush = 1
    elif genital_thrush == 'No':
        genital_thrush = 0

    visual_blurring = st.radio(
        "Apakah anda mengalami pandangan kabur ?", ('Yes', 'No'))
    if visual_blurring == 'Yes':
        visual_blurring = 1
    elif visual_blurring == 'No':
        visual_blurring = 0

    Itching = st.radio("Apakah anda mengalami gatal-gatal ?", ('Yes', 'No'))
    if Itching == 'Yes':
        Itching = 1
    elif Itching == 'No':
        Itching = 0

    Irritability = st.radio(
        "Apakah anda mengalami iritasi ?", ('Yes', 'No'))
    if Irritability == 'Yes':
        Irritability = 1
    elif Irritability == 'No':
        Irritability = 0

    delayed_healing = st.radio(
        "Apakah anda merasa penyembuhan luka berlangsung lama ?", ('Yes', 'No'))
    if delayed_healing == 'Yes':
        delayed_healing = 1
    elif delayed_healing == 'No':
        delayed_healing = 0

    partial_paresis = st.radio(
        "Apakah anda mengalami gangguan gerakan pada sebagian tubuh ?", ('Yes', 'No'))
    if partial_paresis == 'Yes':
        partial_paresis = 1
    elif partial_paresis == 'No':
        partial_paresis = 0

    muscle_stiffness = st.radio(
        "Apakah anda mengalami otot kaku ?", ('Yes', 'No'))
    if muscle_stiffness == 'Yes':
        muscle_stiffness = 1
    elif muscle_stiffness == 'No':
        muscle_stiffness = 0

    Alopecia = st.radio(
        "Apakah anda mengalami kerontokan rambut ?", ('Yes', 'No'))
    if Alopecia == 'Yes':
        Alopecia = 1
    elif Alopecia == 'No':
        Alopecia = 0

    Obesity = st.radio("Apakah anda mengalami obesitas ?", ('Yes', 'No'))
    if Obesity == 'Yes':
        Obesity = 1
    elif Obesity == 'No':
        Obesity = 0

    # input_age = [umur]
    input = [umur, gender, polyuria, polyphagia, sudden_weight_loss, weakness, polyphagia, genital_thrush,
             visual_blurring, Itching, Irritability, delayed_healing, partial_paresis, muscle_stiffness, Alopecia, Obesity]

    X_min = X.min()
    X_max = X.max()

    norm_input = ((input - X_min)/(X_max - X_min))
    norm_input = np.array(norm_input).reshape(1, -1)

    button_hasil = st.button("Cek Hasil")

    if button_hasil:
        from sklearn import tree
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        prediksi_dt = dt.predict(norm_input)
        if prediksi_dt == 0:
            st.success(
                'Hasil prediksi menggunakan decision tree adalah **_Negatif_**')
        elif prediksi_dt == 1:
            st.error(
                'Hasil prediksi menggunakan decision tree adalah **_Positif_**')
