import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tkinter import Tk, Label, Entry, Button, StringVar, Text, END

# Memuat dataset
print("Memuat dataset...")
df = pd.read_csv('games_details.csv', low_memory=False)
print("Dataset berhasil dimuat.")

# Menghapus spasi di nama kolom (jika ada)
df.columns = df.columns.str.strip()

# Menangani missing values jika ada
df = df.dropna(subset=['PTS', 'TEAM_ABBREVIATION'])

# Mengelompokkan data berdasarkan GAME_ID dan TEAM_ABBREVIATION untuk mendapatkan total poin tim
df_team = df.groupby(['GAME_ID', 'TEAM_ABBREVIATION']).agg({'PTS': 'sum'}).reset_index()

# Membuat DataFrame untuk mencocokkan dua tim dalam satu pertandingan
df_matchup = df_team.merge(df_team, on='GAME_ID', suffixes=('_A', '_B'))

# Hanya ambil pasangan tim yang berbeda
df_matchup = df_matchup[df_matchup['TEAM_ABBREVIATION_A'] != df_matchup['TEAM_ABBREVIATION_B']]

# Menentukan pemenang berdasarkan skor
df_matchup['Winner'] = df_matchup.apply(
    lambda row: 1 if row['PTS_A'] > row['PTS_B'] else 0, axis=1
)

# Menambahkan perbedaan skor untuk fitur
df_matchup['Score_diff'] = df_matchup['PTS_A'] - df_matchup['PTS_B']
df_matchup['Total_Points'] = df_matchup['PTS_A'] + df_matchup['PTS_B']
df_matchup['Score_Ratio'] = df_matchup['PTS_A'] / (df_matchup['PTS_B'] + 1e-5)

# Menstandarkan fitur numerik
scaler = StandardScaler()
numerical_features = ['Score_diff', 'Total_Points', 'Score_Ratio']
df_matchup[numerical_features] = scaler.fit_transform(df_matchup[numerical_features])

# Mengencode tim untuk Model
le = LabelEncoder()
df_matchup['Team_A_encoded'] = le.fit_transform(df_matchup['TEAM_ABBREVIATION_A'])
df_matchup['Team_B_encoded'] = le.transform(df_matchup['TEAM_ABBREVIATION_B'])

# Memilih fitur dan target
X = df_matchup[['Team_A_encoded', 'Team_B_encoded'] + numerical_features]
y = df_matchup['Winner']

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Fungsi untuk memprediksi hasil pertandingan
def predict_winner(team_a, team_b):
    original_order = (team_a, team_b)
    if team_a > team_b:
        team_a, team_b = team_b, team_a
        reverse = True
    else:
        reverse = False

    if team_a not in le.classes_ or team_b not in le.classes_:
        valid_teams = list(le.classes_)
        return f"Tim {team_a} atau {team_b} tidak ditemukan dalam data.\nTim yang tersedia: {', '.join(valid_teams)}"

    team_a_encoded = le.transform([team_a])[0]
    team_b_encoded = le.transform([team_b])[0]
    score_diff = 0
    total_points = 0
    score_ratio = 1.0

    input_data = pd.DataFrame([[team_a_encoded, team_b_encoded, score_diff, total_points, score_ratio]],
                               columns=['Team_A_encoded', 'Team_B_encoded', 'Score_diff', 'Total_Points', 'Score_Ratio'])

    prediction = rf_model.predict(input_data)
    prediction_proba = rf_model.predict_proba(input_data)[0]

    if reverse:
        winner = original_order[1] if prediction == 1 else original_order[0]
        winner_proba = prediction_proba[0] * 100
        loser_proba = prediction_proba[1] * 100
    else:
        winner = original_order[0] if prediction == 1 else original_order[1]
        winner_proba = prediction_proba[1] * 100
        loser_proba = prediction_proba[0] * 100

    return winner, winner_proba, loser_proba

# Membuat GUI dengan Tkinter
class NBAApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("NBA Match Predictor")
        self.geometry("500x400")

        # Label dan input untuk Tim A
        Label(self, text="Masukkan nama Tim A:", font=("Arial", 12)).pack(pady=10)
        self.team_a_var = StringVar()
        Entry(self, textvariable=self.team_a_var, font=("Arial", 12)).pack()

        # Label dan input untuk Tim B
        Label(self, text="Masukkan nama Tim B:", font=("Arial", 12)).pack(pady=10)
        self.team_b_var = StringVar()
        Entry(self, textvariable=self.team_b_var, font=("Arial", 12)).pack()

        # Tombol Prediksi
        Button(self, text="Prediksi", font=("Arial", 12), command=self.on_predict).pack(pady=20)

        # Area untuk menampilkan hasil prediksi
        Label(self, text="Hasil Prediksi:", font=("Arial", 12, "bold")).pack(pady=10)
        self.result_text = Text(self, height=8, width=50, font=("Arial", 12), state='disabled')
        self.result_text.pack()

    def on_predict(self):
        team_a = self.team_a_var.get().upper().strip()
        team_b = self.team_b_var.get().upper().strip()

        if not team_a or not team_b:
            self.show_result("Nama tim tidak boleh kosong!")
            return

        result = predict_winner(team_a, team_b)
        if isinstance(result, str):
            self.show_result(result)
        else:
            winner, winner_proba, loser_proba = result
            self.show_result(
                f"Prediksi pertandingan antara {team_a} dan {team_b}:\n"
                f"Tim yang kemungkinan besar menang: {winner}\n"
                f"Persentase kemungkinan {team_a} menang: {winner_proba:.2f}%\n"
                f"Persentase kemungkinan {team_b} menang: {loser_proba:.2f}%"
            )

    def show_result(self, text):
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, END)
        self.result_text.insert(END, text)
        self.result_text.config(state='disabled')

if __name__ == "__main__":
    app = NBAApp()
    app.mainloop()
