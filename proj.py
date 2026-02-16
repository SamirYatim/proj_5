import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Настройка стилей для графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Фиксация случайных значений
np.random.seed(42)
torch.manual_seed(42)

def generate_agri_demand_data(n_days=1000, product='wheat'):
    """
    Генерирует синтетические данные спроса на сельхозпродукцию. Сам датасет на ПК отдельно.
    Включает: тренд, сезонность, шум и события.
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Базовый спрос
    base_demand = 1000
    
    # 1. Тренд (рост спроса со временем)
    trend = np.linspace(0, 200, n_days)
    
    # 2. Сезонность (годовая + недельная)
    days = np.arange(n_days)
    yearly_seasonality = 150 * np.sin(2 * np.pi * days / 365)
    weekly_seasonality = 50 * np.sin(2 * np.pi * days / 7)
    
    # 3. События (праздники, урожай)
    events = np.zeros(n_days)
    # Урожайный сезон (сентябрь-октябрь)
    harvest_months = [9, 10]
    for i, date in enumerate(dates):
        if date.month in harvest_months:
            events[i] = 100
    
    # 4. Шум
    noise = np.random.normal(0, 30, n_days)
    
    # Итоговый спрос
    demand = base_demand + trend + yearly_seasonality + weekly_seasonality + events + noise
    demand = np.maximum(demand, 0)  # Спрос не может быть отрицательным
    
    df = pd.DataFrame({
        'Date': dates,
        'Demand': demand,
        'Product': product
    })
    
    return df

print("Генерация данных о спросе на пшеницу...")
df = generate_agri_demand_data(n_days=1000, product='wheat')
print(f"Всего записей: {len(df)}")
print(f"Период: {df['Date'].min()} - {df['Date'].max()}")
print(df.head())

# =============================================================================
# 2. АНАЛИЗ СЕЗОННОСТИ И ТРЕНДОВ (SEASONALITY ANALYSIS)
# =============================================================================

print("\n" + "="*60)
print("ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА")
print("="*60)

# Декомпозиция для первых 365 дней (для наглядности)
decomposition = seasonal_decompose(df['Demand'].values[:365], model='additive', period=365)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
axes[0].plot(df['Date'][:365], df['Demand'].values[:365], label='Исходный ряд')
axes[0].set_title('Исходный спрос на пшеницу')
axes[0].legend()

axes[1].plot(df['Date'][:365], decomposition.trend[:365], color='red', label='Тренд')
axes[1].set_title('Тренд (долгосрочная динамика)')
axes[1].legend()

axes[2].plot(df['Date'][:365], decomposition.seasonal[:365], color='green', label='Сезонность')
axes[2].set_title('Сезонная компонента (годовая)')
axes[2].legend()

axes[3].plot(df['Date'][:365], decomposition.resid[:365], color='gray', label='Остаток', alpha=0.5)
axes[3].set_title('Остаток (шум)')
axes[3].legend()

plt.tight_layout()
plt.savefig('seasonality_decomposition.png', dpi=150)
plt.show()

# =============================================================================
# 3. ПОДГОТОВКА ДАННЫХ ДЛЯ НЕЙРОСЕТИ (DATA PREPROCESSING)
# =============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback=60, forecast_horizon=30):
        self.data = data.values
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        
        self.X = []
        self.y = []
        
        for i in range(len(self.data) - lookback - forecast_horizon + 1):
            self.X.append(self.data[i : i + lookback])
            self.y.append(self.data[i + lookback : i + lookback + forecast_horizon])
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        # Нормализация
        self.mean = self.X.mean()
        self.std = self.X.std()
        self.X = (self.X - self.mean) / self.std
        self.y = (self.y - self.mean) / self.std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

# Параметры
LOOKBACK = 60      # 60 дней истории
HORIZON = 30       # Прогноз на 30 дней вперед
TRAIN_RATIO = 0.8

dataset = TimeSeriesDataset(df['Demand'], lookback=LOOKBACK, forecast_horizon=HORIZON)
train_size = int(TRAIN_RATIO * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =============================================================================
# 4. МОДЕЛЬ N-BEATS (NEURAL BASELINE EXPONENTIAL SMOOTHING)
# =============================================================================

class NBEATSBlock(nn.Module):
    """Один блок N-BEATS (Backcast + Forecast)"""
    def __init__(self, input_size, hidden_size, forecast_horizon):
        super(NBEATSBlock, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        # Выходы
        self.backcast_fc = nn.Linear(hidden_size, input_size)
        self.forecast_fc = nn.Linear(hidden_size, forecast_horizon)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        # Backcast (восстановление входа) и Forecast (прогноз)
        backcast = self.backcast_fc(x)
        forecast = self.forecast_fc(x)
        
        return backcast, forecast

class NBEATS(nn.Module):
    """
    Архитектура N-BEATS
    Состоит из нескольких блоков, которые последовательно вычитают backcast из входа
    """
    def __init__(self, input_size, hidden_size, forecast_horizon, n_blocks=3):
        super(NBEATS, self).__init__()
        
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, hidden_size, forecast_horizon) 
            for _ in range(n_blocks)
        ])
        
        self.forecast_horizon = forecast_horizon
        
    def forward(self, x):
        # x: (batch, lookback)
        forecast_sum = torch.zeros(x.shape[0], self.forecast_horizon, device=x.device)
        residual = x
        
        for block in self.blocks:
            backcast, forecast = block(residual)
            forecast_sum += forecast
            residual = residual - backcast  # Вычитаем объясненную часть
            
        return forecast_sum

# Инициализация модели
NBEATS_MODEL = NBEATS(
    input_size=LOOKBACK, 
    hidden_size=256, 
    forecast_horizon=HORIZON, 
    n_blocks=4
)

criterion = nn.MSELoss()
optimizer = optim.Adam(NBEATS_MODEL.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# =============================================================================
# 5. ОБУЧЕНИЕ МОДЕЛИ (TRAINING)
# =============================================================================

print("\n" + "="*60)
print("ОБУЧЕНИЕ МОДЕЛИ N-BEATS")
print("="*60)

n_epochs = 50
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    NBEATS_MODEL.train()
    epoch_loss = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = NBEATS_MODEL(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_loss:.6f}')

# График обучения
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title('Динамика функции потерь (N-BEATS)')
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_loss.png', dpi=150)
plt.show()

# =============================================================================
# 6. БАЗОВАЯ МОДЕЛЬ ARIMA (BASELINE)
# =============================================================================

print("\n" + "="*60)
print("ОБУЧЕНИЕ БАЗОВОЙ МОДЕЛИ ARIMA")
print("="*60)

# Для ARIMA берем последние данные для обучения
train_data = df['Demand'].values[:int(len(df) * TRAIN_RATIO)]
test_data = df['Demand'].values[int(len(df) * TRAIN_RATIO):]

# Обучаем ARIMA (p,d,q) = (2,1,2) - типичные параметры для спроса
print("Подбор параметров ARIMA...")
arima_model = ARIMA(train_data, order=(2, 1, 2))
arima_fitted = arima_model.fit()
print(f"ARIMA обучена. AIC: {arima_fitted.aic:.2f}")

# Прогноз ARIMA
arima_forecast = arima_fitted.forecast(steps=HORIZON)

# =============================================================================
# 7. ОЦЕНКА И СРАВНЕНИЕ МОДЕЛЕЙ (EVALUATION)
# =============================================================================

def calculate_metrics(actual, predicted):
    """Расчет всех метрик качества"""
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # SMAPE (Symmetric MAPE)
    smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # MAE
    mae = np.mean(np.abs(actual - predicted))
    
    return {
        'MAPE': mape,
        'SMAPE': smape,
        'RMSE': rmse,
        'MAE': mae
    }

# Тестирование N-BEATS
NBEATS_MODEL.eval()
nbeats_predictions = []
nbeats_actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = NBEATS_MODEL(X_batch)
        # Денормализация
        outputs_denorm = outputs.numpy() * dataset.std + dataset.mean
        y_denorm = y_batch.numpy() * dataset.std + dataset.mean
        
        nbeats_predictions.extend(outputs_denorm.flatten())
        nbeats_actuals.extend(y_denorm.flatten())

nbeats_predictions = np.array(nbeats_predictions)
nbeats_actuals = np.array(nbeats_actuals)

# Тестирование ARIMA (на том же отрезке)
# ARIMA прогнозирует последовательно, берем последние HORIZON точек
arima_predictions = arima_forecast[:len(nbeats_predictions)]
arima_actuals = nbeats_actuals[:len(arima_predictions)]

# Расчет метрик
nbeats_metrics = calculate_metrics(nbeats_actuals, nbeats_predictions)
arima_metrics = calculate_metrics(arima_actuals, arima_predictions)

print("\n" + "="*60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*60)

comparison_df = pd.DataFrame({
    'Метрика': ['MAPE (%)', 'SMAPE (%)', 'RMSE', 'MAE'],
    'N-BEATS': [
        f"{nbeats_metrics['MAPE']:.2f}", 
        f"{nbeats_metrics['SMAPE']:.2f}", 
        f"{nbeats_metrics['RMSE']:.2f}",
        f"{nbeats_metrics['MAE']:.2f}"
    ],
    'ARIMA': [
        f"{arima_metrics['MAPE']:.2f}", 
        f"{arima_metrics['SMAPE']:.2f}", 
        f"{arima_metrics['RMSE']:.2f}",
        f"{arima_metrics['MAE']:.2f}"
    ]
})

print(comparison_df.to_string(index=False))

# Визуализация сравнения метрик
metrics_names = ['MAPE', 'SMAPE', 'RMSE', 'MAE']
nbeats_vals = [nbeats_metrics[m] for m in ['MAPE', 'SMAPE', 'RMSE', 'MAE']]
arima_vals = [arima_metrics[m] for m in ['MAPE', 'SMAPE', 'RMSE', 'MAE']]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, nbeats_vals, width, label='N-BEATS', color='steelblue')
bars2 = ax.bar(x + width/2, arima_vals, width, label='ARIMA', color='coral')

ax.set_ylabel('Значение')
ax.set_title('Сравнение метрик качества прогноза')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Добавление значений на столбцы
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# =============================================================================
# 8. ВИЗУАЛИЗАЦИЯ ПРОГНОЗОВ (FORECAST VISUALIZATION)
# =============================================================================

# Берем последний прогноз для визуализации
last_test_idx = len(test_dataset) - 1
X_last, y_last = test_dataset[last_test_idx]

with torch.no_grad():
    nbeats_last_pred = NBEATS_MODEL(X_last.unsqueeze(0)).numpy().flatten()
    nbeats_last_pred = nbeats_last_pred * dataset.std + dataset.mean

y_last_denorm = y_last.numpy() * dataset.std + dataset.mean

# Индексы для временной оси
history_start = len(df) - LOOKBACK - HORIZON
history_end = len(df) - HORIZON
forecast_start = history_end
forecast_end = forecast_start + HORIZON

dates_history = df['Date'].values[history_start:history_end]
dates_forecast = pd.date_range(start=df['Date'].values[history_end-1], periods=HORIZON+1, freq='D')[1:]

fig, ax = plt.subplots(figsize=(14, 7))

# История
ax.plot(dates_history, df['Demand'].values[history_start:history_end], 
        label='Исторические данные', color='black', linewidth=2)

# Прогноз N-BEATS
ax.plot(dates_forecast, nbeats_last_pred, label='Прогноз N-BEATS', 
        color='steelblue', linewidth=2, linestyle='--')

# Прогноз ARIMA
ax.plot(dates_forecast[:len(arima_predictions)], arima_predictions, 
        label='Прогноз ARIMA', color='coral', linewidth=2, linestyle=':')

# Фактические значения (для теста)
ax.plot(dates_forecast, y_last_denorm, label='Факт (тест)', 
        color='green', linewidth=2, alpha=0.7)

# Область прогноза
ax.axvspan(dates_forecast[0], dates_forecast[-1], alpha=0.2, color='gray', label='Период прогноза')

ax.set_xlabel('Дата')
ax.set_ylabel('Спрос (тонн)')
ax.set_title('Прогноз спроса на пшеницу: N-BEATS vs ARIMA')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('forecast_visualization.png', dpi=150)
plt.show()

# =============================================================================
# 9. АНАЛИЗ ОШИБОК ПО ВРЕМЕНИ (ERROR ANALYSIS)
# =============================================================================

# Распределение ошибок
nbeats_errors = np.abs(nbeats_actuals - nbeats_predictions)
arima_errors = np.abs(arima_actuals - arima_predictions)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Гистограмма ошибок
axes[0].hist(nbeats_errors, bins=30, alpha=0.6, label=f'N-BEATS (MAE: {nbeats_metrics["MAE"]:.1f})', color='steelblue')
axes[0].hist(arima_errors, bins=30, alpha=0.6, label=f'ARIMA (MAE: {arima_metrics["MAE"]:.1f})', color='coral')
axes[0].set_xlabel('Абсолютная ошибка')
axes[0].set_ylabel('Частота')
axes[0].set_title('Распределение ошибок прогноза')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Ошибки по времени (дни прогноза)
error_by_day_nbeats = []
error_by_day_arima = []

for day in range(HORIZON):
    nbeats_day_errors = []
    arima_day_errors = []
    
    for i in range(len(nbeats_predictions) // HORIZON):
        idx = i * HORIZON + day
        if idx < len(nbeats_predictions):
            nbeats_day_errors.append(np.abs(nbeats_actuals[idx] - nbeats_predictions[idx]))
        if idx < len(arima_predictions):
            arima_day_errors.append(np.abs(arima_actuals[idx] - arima_predictions[idx]))
    
    if nbeats_day_errors:
        error_by_day_nbeats.append(np.mean(nbeats_day_errors))
    if arima_day_errors:
        error_by_day_arima.append(np.mean(arima_day_errors))

axes[1].plot(range(1, len(error_by_day_nbeats)+1), error_by_day_nbeats, 
             label='N-BEATS', color='steelblue', marker='o')
axes[1].plot(range(1, len(error_by_day_arima)+1), error_by_day_arima, 
             label='ARIMA', color='coral', marker='s')
axes[1].set_xlabel('День прогноза (горизонт)')
axes[1].set_ylabel('Средняя абсолютная ошибка')
axes[1].set_title('Точность прогноза по горизонту планирования')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150)
plt.show()

# =============================================================================
# 10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ (SAVE RESULTS)
# =============================================================================

results_summary = {
    'Model': ['N-BEATS', 'ARIMA'],
    'MAPE (%)': [nbeats_metrics['MAPE'], arima_metrics['MAPE']],
    'SMAPE (%)': [nbeats_metrics['SMAPE'], arima_metrics['SMAPE']],
    'RMSE': [nbeats_metrics['RMSE'], arima_metrics['RMSE']],
    'MAE': [nbeats_metrics['MAE'], arima_metrics['MAE']],
    'Improvement (%)': [
        0, 
        (arima_metrics['MAPE'] - nbeats_metrics['MAPE']) / arima_metrics['MAPE'] * 100
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('forecast_results.csv', index=False)

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ В forecast_results.csv")
print("="*60)
print(results_df.to_string(index=False))

# Вывод итогового заключения
print("\n" + "="*60)
print("ВЫВОДЫ ПО ПРОЕКТУ")
print("="*60)

improvement = (arima_metrics['MAPE'] - nbeats_metrics['MAPE']) / arima_metrics['MAPE'] * 100

print(f"""
1. Модель N-BEATS показала MAPE: {nbeats_metrics['MAPE']:.2f}%
2. Базовая модель ARIMA показала MAPE: {arima_metrics['MAPE']:.2f}%
3. Улучшение точности: {improvement:.1f}%

4. Ключевые преимущества N-BEATS:
   - Автоматическое выявление сезонности и трендов
   - Лучшая работа с нелинейными зависимостями
   - Масштабируемость на большие объемы данных

5. Рекомендации для production:
   - Использовать скользящее окно для переобучения
   - Добавить внешние признаки (цены, погода, праздники)
   - Реализовать мониторинг дрейфа данных
""")
