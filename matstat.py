import numpy as np
import matplotlib.pyplot as plt
import os


class StatisticsProcessor:
    def __init__(self):
        self.data = None
        self.n = 0
        self.intervals = None
        self.frequencies = None
        self.relative_frequencies = None
        self.grouped_data = None
        self.grouped_frequencies = None
        self.grouped_relative_frequencies = None
        self.interval_bounds = None
        self.midpoints = None
        self.k = 0

    def read_data_from_file(self, filename):
        """Чтение данных из файла с запятой как десятичным разделителем"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                data = []
                for line in file:
                    line = line.strip()
                    if line:
                        # Заменяем запятые на точки для преобразования в float
                        line = line.replace(',', '.')
                        try:
                            value = float(line)
                            data.append(value)
                        except ValueError:
                            # Если строка содержит несколько значений
                            parts = line.split()
                            for part in parts:
                                part = part.replace(',', '.')
                                try:
                                    value = float(part)
                                    data.append(value)
                                except ValueError:
                                    print(f"Предупреждение: значение '{part}' пропущено")

                if len(data) < 100:
                    print(f"Внимание: в файле только {len(data)} значений, рекомендуется не менее 100")

                self.data = np.array(data)
                self.n = len(self.data)
                print(f"✓ Данные успешно загружены. Объем выборки: n = {self.n}")
                print(f"  Минимальное значение: {np.min(self.data):.4f}")
                print(f"  Максимальное значение: {np.max(self.data):.4f}")
                return True

        except FileNotFoundError:
            print(f"✗ Ошибка: файл '{filename}' не найден")
            return False
        except Exception as e:
            print(f"✗ Ошибка при чтении файла: {e}")
            return False

    def set_number_of_intervals(self, k=None):
        """Задание количества интервалов"""
        if self.data is None:
            print("✗ Сначала загрузите данные!")
            return False

        if k is None:
            # Правило Стёрджеса для определения оптимального числа интервалов
            k = int(1 + 3.322 * np.log10(self.n))
            print(f"✓ Рекомендуемое количество интервалов (по правилу Стёрджеса): {k}")
        else:
            k = int(k)

        self.k = k

        # Определение границ интервалов
        data_min = np.min(self.data)
        data_max = np.max(self.data)
        h = (data_max - data_min) / k  # ширина интервала

        # Создание границ интервалов (немного расширяем для красоты)
        self.interval_bounds = np.linspace(data_min - 0.001, data_max + 0.001, k + 1)

        # Вычисление частот
        self.frequencies, _ = np.histogram(self.data, bins=self.interval_bounds)
        self.relative_frequencies = self.frequencies / self.n

        # Вычисление середин интервалов
        self.midpoints = (self.interval_bounds[:-1] + self.interval_bounds[1:]) / 2

        print(f"✓ Количество интервалов: k = {k}")
        print(f"✓ Ширина интервала: h = {h:.4f}")
        print(f"✓ Диапазон данных: [{data_min:.4f}, {data_max:.4f}]")

        print("\nИнтервальный ряд частот:")
        print("-" * 60)
        for i in range(k):
            print(f"[{self.interval_bounds[i]:.4f}, {self.interval_bounds[i + 1]:.4f}): {self.frequencies[i]:3d}")

        print("\nИнтервальный ряд относительных частот:")
        print("-" * 60)
        for i in range(k):
            print(
                f"[{self.interval_bounds[i]:.4f}, {self.interval_bounds[i + 1]:.4f}): {self.relative_frequencies[i]:.4f}")

        return True

    def create_grouped_series(self):
        """Создание группированного ряда"""
        if self.frequencies is None:
            print("✗ Сначала создайте интервальный ряд!")
            return False

        # Группированный ряд строится по серединам интервалов
        self.grouped_frequencies = self.frequencies.copy()
        self.grouped_relative_frequencies = self.relative_frequencies.copy()
        self.grouped_data = self.midpoints

        print("\n✓ Группированный ряд частот (по серединам интервалов):")
        print("-" * 60)
        for i in range(self.k):
            print(f"x_{i + 1:2d} = {self.midpoints[i]:.4f}: {self.grouped_frequencies[i]:3d}")

        print("\n✓ Группированный ряд относительных частот:")
        print("-" * 60)
        for i in range(self.k):
            print(f"x_{i + 1:2d} = {self.midpoints[i]:.4f}: {self.grouped_relative_frequencies[i]:.4f}")

        return True

    def plot_histograms(self):
        """Построение гистограмм частот и относительных частот"""
        if self.frequencies is None:
            print("✗ Сначала создайте интервальный ряд!")
            return False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Ширина интервалов
        widths = np.diff(self.interval_bounds)

        # Гистограмма частот
        max_height1 = 0
        for i in range(self.k):
            # Высота = частота / ширина интервала (чтобы площадь = частоте)
            height = self.frequencies[i] / widths[i] if widths[i] > 0 else 0
            if height > max_height1:
                max_height1 = height

        for i in range(self.k):
            height = self.frequencies[i] / widths[i] if widths[i] > 0 else 0
            rect = ax1.bar(self.interval_bounds[i], height, width=widths[i],
                           align='edge', edgecolor='black', alpha=0.7,
                           color=plt.cm.Blues(0.5 + 0.3 * i / self.k))

            # Добавляем площадь внутрь прямоугольника
            if height > 0:
                area = self.frequencies[i]  # площадь = частота
                # Позиция текста - центр прямоугольника
                text_x = self.interval_bounds[i] + widths[i] / 2
                text_y = height / 2
                ax1.text(text_x, text_y, f'{area:.0f}',
                         ha='center', va='center', fontweight='bold',
                         color='white', fontsize=9)

        ax1.set_xlabel('Значения')
        ax1.set_ylabel('Плотность частоты (частота/шаг)')
        ax1.set_title('Гистограмма частот\n(площадь прямоугольника = частота)')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(self.interval_bounds[0], self.interval_bounds[-1])
        ax1.set_ylim(0, max_height1 * 1.1)  # Начинаем с 0
        ax1.axhline(y=0, color='black', linewidth=0.8)

        # Гистограмма относительных частот
        max_height2 = 0
        for i in range(self.k):
            # Высота = относительная частота / ширина интервала
            height = self.relative_frequencies[i] / widths[i] if widths[i] > 0 else 0
            if height > max_height2:
                max_height2 = height

        for i in range(self.k):
            height = self.relative_frequencies[i] / widths[i] if widths[i] > 0 else 0
            rect = ax2.bar(self.interval_bounds[i], height, width=widths[i],
                           align='edge', edgecolor='black', alpha=0.7,
                           color=plt.cm.Reds(0.5 + 0.3 * i / self.k))

            # Добавляем площадь внутрь прямоугольника
            if height > 0:
                area = self.relative_frequencies[i]  # площадь = относительная частота
                # Позиция текста - центр прямоугольника
                text_x = self.interval_bounds[i] + widths[i] / 2
                text_y = height / 2
                ax2.text(text_x, text_y, f'{area:.3f}',
                         ha='center', va='center', fontweight='bold',
                         color='white', fontsize=9)

        ax2.set_xlabel('Значения')
        ax2.set_ylabel('Плотность относительной частоты')
        ax2.set_title('Гистограмма относительных частот\n(площадь прямоугольника = относительная частота)')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(self.interval_bounds[0], self.interval_bounds[-1])
        ax2.set_ylim(0, max_height2 * 1.1)  # Начинаем с 0
        ax2.axhline(y=0, color='black', linewidth=0.8)

        plt.tight_layout()
        plt.show()
        print("✓ Гистограммы построены")

    def plot_polygons(self):
        """Построение полигонов для группированного ряда"""
        if self.grouped_frequencies is None:
            print("✗ Сначала создайте группированный ряд!")
            return False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Полигон частот
        ax1.plot(self.midpoints, self.grouped_frequencies, 'bo-', linewidth=2, markersize=8)
        ax1.fill_between(self.midpoints, self.grouped_frequencies, alpha=0.3, color='blue')

        # Добавляем точки к началу и концу для замыкания
        x_poly = np.concatenate([[self.midpoints[0] - (self.midpoints[1] - self.midpoints[0]) / 2],
                                 self.midpoints,
                                 [self.midpoints[-1] + (self.midpoints[-1] - self.midpoints[-2]) / 2]])
        y_poly = np.concatenate([[0], self.grouped_frequencies, [0]])

        ax1.set_xlabel('Середины интервалов')
        ax1.set_ylabel('Частота')
        ax1.set_title('Полигон частот')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(x_poly[0], x_poly[-1])
        ax1.set_ylim(0, max(self.grouped_frequencies) * 1.1)

        # Добавляем значения на точки
        for i, (x, y) in enumerate(zip(self.midpoints, self.grouped_frequencies)):
            ax1.text(x, y + 0.2, f'{y}', ha='center', va='bottom', fontsize=9)

        # Полигон относительных частот
        ax2.plot(self.midpoints, self.grouped_relative_frequencies, 'ro-', linewidth=2, markersize=8)
        ax2.fill_between(self.midpoints, self.grouped_relative_frequencies, alpha=0.3, color='red')
        ax2.set_xlabel('Середины интервалов')
        ax2.set_ylabel('Относительная частота')
        ax2.set_title('Полигон относительных частот')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(x_poly[0], x_poly[-1])
        ax2.set_ylim(0, max(self.grouped_relative_frequencies) * 1.1)

        # Добавляем значения на точки
        for i, (x, y) in enumerate(zip(self.midpoints, self.grouped_relative_frequencies)):
            ax2.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()
        print("✓ Полигоны построены")

    def empirical_distribution_function(self, x, series_type='interval'):
        """Эмпирическая функция распределения"""
        if self.data is None:
            return None

        if series_type == 'interval':
            # Для интервального ряда
            sorted_data = np.sort(self.data)
            n = len(sorted_data)
            result = np.zeros_like(x, dtype=float)

            for i, val in enumerate(x):
                result[i] = np.sum(sorted_data <= val) / n

            return result

        elif series_type == 'grouped':
            # Для группированного ряда
            if self.grouped_relative_frequencies is None:
                return None

            cumulative_freq = np.cumsum(self.grouped_relative_frequencies)
            result = np.zeros_like(x, dtype=float)

            for i, val in enumerate(x):
                # Находим индекс интервала, в который попадает val
                idx = np.searchsorted(self.midpoints, val, side='right')
                if idx == 0:
                    result[i] = 0
                elif idx >= len(cumulative_freq):
                    result[i] = 1.0
                else:
                    result[i] = cumulative_freq[idx - 1]

            return result

    def plot_empirical_functions(self):
        """Построение графиков эмпирических функций распределения"""
        if self.data is None:
            print("✗ Сначала загрузите данные!")
            return False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Для интервального ряда
        x_min = np.min(self.data) - 0.1
        x_max = np.max(self.data) + 0.1
        x_vals = np.linspace(x_min, x_max, 1000)
        y_interval = self.empirical_distribution_function(x_vals, 'interval')

        # Для правильного отображения ступенчатой функции
        sorted_data = np.sort(self.data)
        x_step = np.concatenate([[x_min], sorted_data, [x_max]])
        y_step = np.concatenate([[0], np.arange(1, len(self.data) + 1) / len(self.data), [1]])

        ax1.step(x_step, y_step, where='post', linewidth=2, color='blue')
        ax1.set_xlabel('x')
        ax1.set_ylabel('F*(x)')
        ax1.set_title('Эмпирическая функция распределения\n(для интервального ряда)')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlim(x_min, x_max)

        # Для группированного ряда
        if self.grouped_relative_frequencies is not None:
            cumulative_freq = np.cumsum(self.grouped_relative_frequencies)

            # Создаем ступенчатую функцию
            x_points = np.concatenate([[self.midpoints[0] - 0.1], self.midpoints, [self.midpoints[-1] + 0.1]])
            y_points = np.concatenate([[0], cumulative_freq, [1]])

            ax2.step(x_points, y_points, where='post', linewidth=2, color='red')
            ax2.set_xlabel('x')
            ax2.set_ylabel('F*(x)')
            ax2.set_title('Эмпирическая функция распределения\n(для группированного ряда)')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xlim(x_min, x_max)

            # Добавляем точки разрыва
            for i, (x, y) in enumerate(zip(self.midpoints, cumulative_freq)):
                ax2.plot(x, y, 'ro', markersize=6)
                ax2.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=8)

        plt.tight_layout()
        plt.show()
        print("✓ Эмпирические функции распределения построены")

    def calculate_statistics(self):
        """Вычисление числовых характеристик выборки"""
        if self.data is None:
            print("✗ Сначала загрузите данные!")
            return None

        print("\n" + "=" * 70)
        print("ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ ВЫБОРКИ")
        print("=" * 70)

        # Формулы для вычислений
        print("\nФОРМУЛЫ ДЛЯ ВЫЧИСЛЕНИЯ:")
        print("1. Выборочное среднее: x̄ = (1/n) * Σx_i")
        print("2. Выборочная дисперсия: D_в = (1/n) * Σ(x_i - x̄)^2")
        print("3. Исправленная дисперсия: S² = (1/(n-1)) * Σ(x_i - x̄)^2")
        print("4. Выборочное СКО: σ_в = √D_в")
        print("5. Исправленное СКО: S = √S²")
        print("\n" + "-" * 70)

        # Вычисления
        n = self.n
        x_mean = np.mean(self.data)  # x̄

        # Выборочная дисперсия
        D_v = np.var(self.data, ddof=0)  # D_в

        # Исправленная дисперсия
        S2 = np.var(self.data, ddof=1)  # S²

        # Выборочное СКО
        sigma_v = np.sqrt(D_v)  # σ_в

        # Исправленное СКО
        S = np.sqrt(S2)  # S

        # Мода (используем numpy для вычисления)
        values, counts = np.unique(self.data, return_counts=True)
        max_count_index = np.argmax(counts)
        mode_val = values[max_count_index]
        mode_count = counts[max_count_index]

        # Дополнительные характеристики
        median = np.median(self.data)  # Медиана

        # Квартили
        Q1 = np.percentile(self.data, 25)
        Q3 = np.percentile(self.data, 75)
        IQR = Q3 - Q1

        # Коэффициенты
        CV = (sigma_v / x_mean) * 100 if x_mean != 0 else 0  # Коэффициент вариации (%)

        # Коэффициент асимметрии
        n = len(self.data)
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)  # исправленное СКО
        if std > 0:
            skewness = (1 / n) * np.sum(((self.data - mean) / std) ** 3)
        else:
            skewness = 0

        # Коэффициент эксцесса
        if std > 0 and n > 3:
            kurtosis = (1 / n) * np.sum(((self.data - mean) / std) ** 4) - 3
        else:
            kurtosis = 0

        print("\n✓ РЕЗУЛЬТАТЫ:")
        print(f"  Объем выборки: n = {n}")
        print(f"  1. Выборочное среднее (x̄) = {x_mean:.6f}")
        print(f"  2. Выборочная дисперсия (D_в) = {D_v:.6f}")
        print(f"  3. Исправленная дисперсия (S²) = {S2:.6f}")
        print(f"  4. Выборочное СКО (σ_в) = {sigma_v:.6f}")
        print(f"  5. Исправленное СКО (S) = {S:.6f}")

        print("\n✓ ДОПОЛНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ:")
        print(f"  Медиана = {median:.6f}")
        print(f"  Мода = {mode_val:.6f} (встречается {mode_count} раз)")
        print(f"  Минимум = {np.min(self.data):.6f}")
        print(f"  Максимум = {np.max(self.data):.6f}")
        print(f"  Размах = {np.max(self.data) - np.min(self.data):.6f}")
        print(f"  Q1 (25-й процентиль) = {Q1:.6f}")
        print(f"  Q3 (75-й процентиль) = {Q3:.6f}")
        print(f"  Межквартильный размах (IQR) = {IQR:.6f}")
        print(f"  Коэффициент вариации = {CV:.2f}%")
        print(f"  Коэффициент асимметрии = {skewness:.6f}")
        print(f"  Коэффициент эксцесса = {kurtosis:.6f}")

        return {
            'n': n, 'x_mean': x_mean, 'D_v': D_v, 'S2': S2,
            'sigma_v': sigma_v, 'S': S, 'median': median,
            'mode': mode_val, 'min': np.min(self.data),
            'max': np.max(self.data), 'range': np.max(self.data) - np.min(self.data),
            'Q1': Q1, 'Q3': Q3, 'IQR': IQR, 'CV': CV,
            'skewness': skewness, 'kurtosis': kurtosis
        }

    def run_all_automatically(self, filename="20.txt"):
        """Автоматическое выполнение всех шагов"""
        print("\n" + "=" * 70)
        print("АВТОМАТИЧЕСКОЕ ВЫПОЛНЕНИЕ ВСЕХ ШАГОВ")
        print("=" * 70)

        # Шаг 1: Загрузка данных
        print("\n1. ЗАГРУЗКА ДАННЫХ:")
        if not self.read_data_from_file(filename):
            return False

        # Шаг 2: Создание интервального ряда
        print("\n2. СОЗДАНИЕ ИНТЕРВАЛЬНОГО РЯДА:")
        if not self.set_number_of_intervals():
            return False

        # Шаг 3: Создание группированного ряда
        print("\n3. СОЗДАНИЕ ГРУППИРОВАННОГО РЯДА:")
        if not self.create_grouped_series():
            return False

        # Шаг 4: Вычисление статистик
        print("\n4. ВЫЧИСЛЕНИЕ ЧИСЛОВЫХ ХАРАКТЕРИСТИК:")
        stats_result = self.calculate_statistics()

        # Шаг 5: Построение гистограмм
        print("\n5. ПОСТРОЕНИЕ ГИСТОГРАММ:")
        self.plot_histograms()

        # Шаг 6: Построение полигонов
        print("\n6. ПОСТРОЕНИЕ ПОЛИГОНОВ:")
        self.plot_polygons()

        # Шаг 7: Построение эмпирических функций распределения
        print("\n7. ПОСТРОЕНИЕ ЭМПИРИЧЕСКИХ ФУНКЦИЙ РАСПРЕДЕЛЕНИЯ:")
        self.plot_empirical_functions()

        # Шаг 8: Сохранение результатов
        print("\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ:")
        self.save_results_to_file("automatic_results.txt", stats_result)

        print("\n" + "=" * 70)
        print("✓ ВСЕ ШАГИ УСПЕШНО ВЫПОЛНЕНЫ!")
        print("=" * 70)

        return True

    def save_results_to_file(self, filename="results.txt", stats_result=None):
        """Сохранение результатов в файл"""
        if self.data is None:
            print("✗ Нет данных для сохранения!")
            return False

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОЙ ОБРАБОТКИ\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Исходный файл: 20.txt\n")
                f.write(f"Объем выборки: n = {self.n}\n\n")

                if self.frequencies is not None:
                    f.write("ИНТЕРВАЛЬНЫЙ РЯД:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Количество интервалов: k = {self.k}\n")
                    f.write(f"Ширина интервала: h = {np.diff(self.interval_bounds)[0]:.4f}\n\n")

                    f.write("Интервалы и частоты:\n")
                    for i in range(self.k):
                        f.write(f"[{self.interval_bounds[i]:.4f}, {self.interval_bounds[i + 1]:.4f}): "
                                f"частота = {self.frequencies[i]:3d}, "
                                f"отн. частота = {self.relative_frequencies[i]:.4f}\n")

                    f.write("\nГРУППИРОВАННЫЙ РЯД:\n")
                    f.write("-" * 70 + "\n")
                    for i in range(self.k):
                        f.write(f"x_{i + 1:2d} = {self.midpoints[i]:.4f}: "
                                f"частота = {self.grouped_frequencies[i]:3d}, "
                                f"отн. частота = {self.grouped_relative_frequencies[i]:.4f}\n")

                # Если статистики не переданы, вычисляем их
                if stats_result is None:
                    stats_result = self.calculate_statistics()

                f.write("\nЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Выборочное среднее (x̄) = {stats_result['x_mean']:.6f}\n")
                f.write(f"Выборочная дисперсия (D_в) = {stats_result['D_v']:.6f}\n")
                f.write(f"Исправленная дисперсия (S²) = {stats_result['S2']:.6f}\n")
                f.write(f"Выборочное СКО (σ_в) = {stats_result['sigma_v']:.6f}\n")
                f.write(f"Исправленное СКО (S) = {stats_result['S']:.6f}\n")
                f.write(f"Медиана = {stats_result['median']:.6f}\n")
                f.write(f"Мода = {stats_result['mode']:.6f}\n")
                f.write(f"Минимум = {stats_result['min']:.6f}\n")
                f.write(f"Максимум = {stats_result['max']:.6f}\n")
                f.write(f"Размах = {stats_result['range']:.6f}\n")
                f.write(f"Коэффициент вариации = {stats_result['CV']:.2f}%\n")
                f.write(f"Коэффициент асимметрии = {stats_result['skewness']:.6f}\n")
                f.write(f"Коэффициент эксцесса = {stats_result['kurtosis']:.6f}\n")

            print(f"✓ Результаты сохранены в файл '{filename}'")
            return True

        except Exception as e:
            print(f"✗ Ошибка при сохранении: {e}")
            return False

    def print_menu(self):
        """Вывод меню программы"""
        print("\n" + "=" * 60)
        print("ПРОГРАММА ПЕРВИЧНОЙ ОБРАБОТКИ СТАТИСТИЧЕСКИХ ДАННЫХ")
        print("=" * 60)
        print("1. Загрузить данные из файла")
        print("2. Задать количество интервалов и создать интервальный ряд")
        print("3. Создать группированный ряд")
        print("4. Построить гистограммы (интервальный ряд)")
        print("5. Построить полигоны (группированный ряд)")
        print("6. Построить эмпирические функции распределения")
        print("7. Вычислить числовые характеристики")
        print("8. Показать все графики вместе")
        print("9. Сохранить результаты в файл")
        print("10. ВЫПОЛНИТЬ ВСЕ ШАГИ АВТОМАТИЧЕСКИ")
        print("0. Выход")
        print("=" * 60)


def create_data_file_20(filename="20.txt"):
    """Создание файла с данными из условия"""
    data = """1,5 1,2 1,5 1,3 1,4 1,6 1,2 1,5 1,5 1,4
1,1 1,5 0,9 1,6 1,2 1,3 1,6 1,5 1,1 1,5
1,0 1,2 1,2 1,0 1,4 1,8 1,3 1,4 1,7 1,3
1,4 1,6 0,9 1,6 1,3 1,1 1,6 1,1 1,4 1,8
1,1 1,3 1,6 1,1 1,5 1,4 1,1 1,7 1,8 1,1
1,5 1,0 1,3 1,4 1,2 1,4 1,8 1,3 1,2 1,5
1,3 1,4 1,1 1,3 1,6 1,3 1,1 1,4 1,3 1,2
1,6 1,1 1,5 1,1 1,3 1,6 1,4 1,6 1,4 1,2
1,3 1,1 1,3 1,6 1,2 1,4 1,4 1,2 1,5 1,7
1,2 1,4 1,1 1,3 1,3 1,2 1,4 1,3 1,2 1,4"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(data)

    print(f"✓ Создан файл '{filename}' с данными из условия (100 значений)")


def main():
    processor = StatisticsProcessor()

    # Создание файла 20.txt с данными из условия, если его нет
    if not os.path.exists("20.txt"):
        create_data_file_20("20.txt")

    print("\n" + "=" * 60)
    print("ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ ПРОГРАММЫ")
    print("=" * 60)
    print("✓ Для быстрого выполнения всех задач выберите пункт 10")
    print("✓ Программа создаст файл '20.txt' с вашими данными")
    print("✓ Десятичный разделитель - запятая (автоматическая замена на точку)")
    print("✓ Объем выборки: 100 значений")
    print("✓ Для построения графиков установите: pip install numpy matplotlib")
    print("=" * 60)

    while True:
        processor.print_menu()
        choice = input("\nВыберите пункт меню (0-10): ").strip()

        if choice == '1':
            filename = input("Введите имя файла с данными (по умолчанию: 20.txt): ").strip()
            if not filename:
                filename = "20.txt"
            processor.read_data_from_file(filename)

        elif choice == '2':
            if processor.data is not None:
                k_input = input("Введите количество интервалов (Enter для автоматического выбора): ").strip()
                if k_input:
                    try:
                        k = int(k_input)
                        if k < 2:
                            print("✗ Количество интервалов должно быть не менее 2!")
                            continue
                        processor.set_number_of_intervals(k)
                    except ValueError:
                        print("✗ Ошибка: введите целое число!")
                else:
                    processor.set_number_of_intervals()
            else:
                print("✗ Сначала загрузите данные (пункт 1)!")

        elif choice == '3':
            processor.create_grouped_series()

        elif choice == '4':
            processor.plot_histograms()

        elif choice == '5':
            processor.plot_polygons()

        elif choice == '6':
            processor.plot_empirical_functions()

        elif choice == '7':
            processor.calculate_statistics()

        elif choice == '8':
            # Показать все графики
            if processor.data is not None:
                if processor.frequencies is not None:
                    processor.plot_histograms()
                    processor.plot_polygons()
                    processor.plot_empirical_functions()
                else:
                    print("✗ Сначала создайте интервальный ряд (пункт 2)!")
            else:
                print("✗ Сначала загрузите данные (пункт 1)!")

        elif choice == '9':
            filename = input("Введите имя файла для сохранения (по умолчанию: results.txt): ").strip()
            if not filename:
                filename = "results.txt"
            processor.save_results_to_file(filename)

        elif choice == '10':
            filename = input("Введите имя файла с данными (по умолчанию: 20.txt): ").strip()
            if not filename:
                filename = "20.txt"
            processor.run_all_automatically(filename)

        elif choice == '0':
            print("\nВыход из программы. До свидания!")
            break

        else:
            print("✗ Неверный выбор. Попробуйте снова.")

        input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    main()