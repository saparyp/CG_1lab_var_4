# Лабораторная работа: Усеченный правильный тетраэдр (Вариант 4)

## Начало работы

### 1. Подготовка репозитория

Клонируйте репозиторий или убедитесь, что вы находитесь в корневой папке проекта:
```bash
git clone https://github.com/saparyp/CG_1lab_var_4
cd CG_1lab_var_4
```

### 2. Конфигурация проекта

Выполните одну из следующих команд CMake для загрузки зависимостей и генерации файлов сборки. Выберите пресет, соответствующий вашему компилятору:

```bash
cmake --preset msvc-debug  # для Windows (Visual Studio / MSVC)
cmake --preset mingw-debug # для Windows (MinGW)
cmake --preset debug       # для GNU/Linux (GCC/Clang)
```

### 3. Сборка проекта

Для сборки используйте команду ниже. Если вы выбрали пресет отладки, файлы будут помещены в директорию `build-debug`.

```bash
cmake --build build-debug --parallel
```

### 4. Запуск приложения

Запустите исполняемый файл из корня проекта:

```bash
# Для сборок MSVC (Visual Studio):
./build-debug/Debug/vulkan-starter-app.exe

# Для сборок MinGW:
./build-debug/vulkan-starter-app.exe
```
