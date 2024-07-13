# Proyectos Finales Machine Learning: Análisis de Sentimiento

## 1. Instalar git
Descargar git para su Sistema Operativo (OS): https://git-scm.com/downloads

## 2. Clonar el repositorio
Necesitará asociar su cuenta de GitHub a su computadora. Puede hacerlo por HTTPS o SSH.
- **HTTPS**: Le pedirán su usuario y contraseña de GitHub al momento de clonar el repositorio.
- **SSH**: Necesitará configurar una llave SSH en su cuenta de GitHub. Siga los pasos de la documentación oficial de GitHub: https://docs.github.com/es/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

Una vez haya clonado el repositorio, podrá acceder a la carpeta del proyecto final y contribuir a él. Por defecto, la rama principal es `main`.

## 3. Contribuir al repositorio
Hay algunas consideraciones a tener en cuenta al momento de contribuir al repositorio.

### 3.1. La rama `main` es protegida
No se puede hacer push directamente a ella. Esta rama contendrá las versiones de los documentos finales y sin errores para asegurar que, si alguien intenta ejecutar los documentos, no tendrá problemas.

### 3.2. Flujo de trabajo
- **Actualizar tu rama local `main` con la rama remota `main`.**
  ```bash
  git switch main # Cambiar a la rama main
  git pull origin main # Actualizar la rama main
    ```
- **Crear una nueva rama a partir de la rama `main`.** El nombre de la rama debe ser descriptivo de los cambios que se harán. Por ejemplo feature/AGREGAR-ANALISIS-DESCRIPTIVO
    ```bash
    git switch -c nombre-rama # Crear y cambiar a la nueva rama
    ```
- **Hacer los cambios necesarios en la rama creada.**
- **Hacer un commit con los cambios realizados.** El mensaje del commit debe ser descriptivo de los cambios realizados.
    ```bash
    git add . # Agregar los cambios
    git commit -m "Mensaje del commit" # Hacer el commit
    ```
- **Actualizar la rama local con la rama remota `main`.**
- **Actualizar la rama creada con la rama `main`.**
    ```bash
    git switch main # Cambiar a la rama main
    git pull origin main # Actualizar la rama main
    git switch nombre-rama # Cambiar a la rama creada
    git merge main # Actualizar la rama creada
    ```
- **Resolver los conflictos si los hay.**
- **Hacer un push de la rama al repositorio remoto.**
    ```bash
    git push origin nombre-rama # Hacer push de la rama creada
    ```
- **Crear un Pull Request (PR) a la rama `main`.**
- **Esperar a que alguien revise los cambios y apruebe el PR.**
- **Una vez aprobado el PR, se podrá hacer merge a la rama `main`.**

**Nota: Una vez tenga el repositorio clonado, descomprimir el archivo `dataset.zip` y nombrar al archivo `dataset.csv` para poder manejar todos el mismo nombre del archivo.**