# project_WDSI
System wykrywający znaki ograniczeń prędkości na zdjęciach.


Wyjaśnienie działania poszczególnych funkcji:
LoadAndSplit - funkcja służy do załadowania a następnie podzielenia zdjęć oraz plików (.xml) zawierających informacje o zdjęciach na 2 zbiory (treningowy i testowy). Funkcja przyjmuje 2 argumenty (pierwszy to ścieżka do plików annotations (.xml), drugi to ścieżka do zdjęć). Pierwszym krokiem funkcji jest wczytanie wsztstkich plików annotations oraz zapisania ich nazwy, nazwy zdjęcia do którego się odnoszą oraz ich klasyID ('speedlimit' albo 'other'). Ilość wystąpień poszczególnych klassID jest zliczana a następnie zbiór jest dzielony w proporcji 3:1 pomiędzy zbiór treningowy i testowy zachowując tą samą proporcję pomiędzy znakami ograniczenia prędkości a innymi znakami. Uwaga: pliki są kopiowane do nadrzędnych folderów train/images, train/annotations oraz test/images, test/annotations nie przenoszone. 

load_ClassIdAndCropPhoto - funkcja zwraca listę słowników w następującym formacie: wczytane zdjęcie pod kluczem 'image', zapisana klassaID/etykieta (identyfikator zdjęcia 'speedlimit' lub 'other) pod kluczem 'labels'. Argumentami funkcji jest ścieżka do folderu annotations oraz do folderu ze zdjęciami. Funkcja wczytuje informacje o rodzaju znaku na zdjęciu oraz jego umiejscowieniu, następnie wycina odpowiedni fragment zdjęcia oraz przypisując mu odpowiednią etykietę umiejscawia w liście.

boVW - funkcja wykorzystując algorytm boVW wykrywa kluczowe punkty dla każdego obrazu osobno oraz zapisuje je w postaci deskryptorów. Następnie tworzony jest słownik wizualny poprzez klasteryzację wszystkich deskryptorów z wykorzystaniem algorytmu K-Means. Argumentem funkcji jest lista słowników zawierająca pod kluczem 'image' zdjęcie (w naszym przypadku fragment zdjęcia) oraz przypisaną mu etykiete. Zwracany jest słownik wizualny.

extractingFeatures - funkcja przyjmuje jako argument zbudowany wcześniej słownik wizualny oraz listę słowników zawierającą zdjęcia oraz ich etykiety. Funkcja oblicza deskryptory obrazu a następnie przypisuje każdemu najbliższe słowo ze słownika wizualnym oraz zlicza częstotliwość ich wystąpień na obrazie (tworzy znormalizowany histogram wystąpienia danych słów). Tym sposobem powstaje wektor cech obrazu. Następnie uzyskany wszystkie wektory cech obrazu ze wszystkich obrazów zapisywane są w liście słowników w następującej formule: wektor cech obrazu/ histogram pod kluczem 'data' oraz etykieta zdjęcia dla którego ten histogram był tworzony jest przepisywana pod kluczem 'label'

train - argumentem funkcji jest lista słowników zawierająca wyodrębniony dla każdego zdjęcia jego wektor cech oraz odpowiednia etykieta. Wykorzystując algorytm drzew decyzyjnych, odpowiednie wetkrory cech są klasyfikowane pomiędzy odpowiednimi etykietami. Zwracany jest wytrenowany model.

predict - funcja odpowida za przewidywanie odpowiedniej etykiety na podstawie wektora cech obrazu oraz wytrenowanego modelu.

claassification - funkcja w pierwszym kroku wczytuję następujące informacje: liczba plików do przetworzenia, następnie dla wszystkich plików wczytuje nazwę pliku oraz ilość wycinków orazu do sklasyfikowania. Następnie w pętli dla każdego wycinka program pobiera współrzędne poszczególnych wycinków. Wykorzystując pobrane współrzędne, wczytuję odpowiedni fragment zdjęcia o nazwie podanej wcześniej z folderu 'images' znajdującym się w katalogu 'test'. Wczytane fragmenty zdjęć są zapisywane do listy słowników, gdzie kluczem jest 'image', etykieta jest natomiast nieznana dlatego zapisywana jest w formacie 'None' pod kluczem 'label'. Funkcja wykorzystując podany w jej argumencie słownik wizualny, korzystając z funkcji extractingFeautres tworzy wektor cech obrazu dla wszystkich wczytanych fragmentów zdjęc. Następnie wykorzystując funkcję predict wyświetla wyniki klasyfikacji odpowiednich fragmentów zdjęc w kolejności ich wczytywania.

Funkcję dodatkowe znajdjące się pod programem podstawowym służyły do badania oraz analizy jakości programu rozpoznającego znaki ograniczenia prędkości. Nie zostały one usunięte ze względu na możliwą potrzebę ich późniejszego wykorzystania. Przydatnymi funkcjami w ocenie klasyfikacji będącej jedną z 2 częsci realizowanego projektu są readText - wczytująca plik 'samples.txt', compareData - porównująca przy pomocny funkcji accuracyScore wynik klasyfikacji ze wzorową klasyfikacją. W celu skorzystania z wyżej wymienionych funkcji należy przenieść pod deklaracje bibliotek oraz odkomentować odpowiednią linijkę w funkcji classification.
Podczas testu oraz porównania klasyfikacji przy wykorzystaniu plików 'samples.txt' jako wzorowy oraz 'input.txt' uzyskano 96% skuteczność klasyfikacji.
