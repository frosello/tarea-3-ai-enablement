"""
Utilidades para cargar y procesar documentos.

Este módulo contiene funciones para:
- Cargar diferentes tipos de documentos (TXT, PDF, DOCX, DOC)
- Procesar y limpiar texto
- Manejar errores de codificación
- Extraer metadatos de archivos
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configurar logging
logger = logging.getLogger(__name__)


def load_text_file(file_path: str) -> str:
    """
    Carga un archivo de texto plano.
    
    Args:
        file_path (str): Ruta del archivo de texto
        
    Returns:
        str: Contenido del archivo
        
    Raises:
        ValueError: Si no se puede decodificar el archivo
        FileNotFoundError: Si el archivo no existe
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Intentar con diferentes codificaciones
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    logger.warning(f"Archivo {file_path} cargado con codificación {encoding}")
                    return content
            except UnicodeDecodeError:
                continue
        raise ValueError(f"No se pudo decodificar el archivo {file_path}")
    except Exception as e:
        logger.error(f"Error cargando archivo de texto {file_path}: {e}")
        raise


def load_pdf_file(file_path: str) -> str:
    """
    Carga un archivo PDF usando PyPDF2.
    
    Args:
        file_path (str): Ruta del archivo PDF
        
    Returns:
        str: Texto extraído del PDF
        
    Raises:
        ImportError: Si PyPDF2 no está instalado
        ValueError: Si no se puede extraer texto del PDF
    """
    try:
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(f"\n--- Página {page_num + 1} ---\n{text}")
                except Exception as e:
                    logger.warning(f"Error extrayendo texto de la página {page_num + 1}: {e}")
                    continue
            
            if not text_content:
                raise ValueError("No se pudo extraer texto del PDF")
            
            full_text = "\n".join(text_content)
            logger.info(f"PDF {file_path} procesado: {len(pdf_reader.pages)} páginas")
            return full_text
            
    except ImportError:
        raise ImportError("PyPDF2 no está instalado. Instala con: pip install PyPDF2")
    except Exception as e:
        logger.error(f"Error cargando archivo PDF {file_path}: {e}")
        raise


def load_docx_file(file_path: str) -> str:
    """
    Carga un archivo DOCX usando python-docx.
    
    Args:
        file_path (str): Ruta del archivo DOCX
        
    Returns:
        str: Texto extraído del documento
        
    Raises:
        ImportError: Si python-docx no está instalado
        ValueError: Si no se encuentra texto en el documento
    """
    try:
        from docx import Document
        
        doc = Document(file_path)
        paragraphs = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text)
        
        if not paragraphs:
            raise ValueError("No se encontró texto en el documento DOCX")
        
        text = "\n\n".join(paragraphs)
        logger.info(f"DOCX {file_path} procesado: {len(paragraphs)} párrafos")
        return text
        
    except ImportError:
        raise ImportError("python-docx no está instalado. Instala con: pip install python-docx")
    except Exception as e:
        logger.error(f"Error cargando archivo DOCX {file_path}: {e}")
        raise


def load_csv_file(file_path: str) -> str:
    """
    Carga un archivo CSV y lo convierte a texto estructurado.
    
    Args:
        file_path (str): Ruta del archivo CSV
        
    Returns:
        str: Contenido del CSV formateado como texto
        
    Raises:
        ImportError: Si pandas no está instalado
        ValueError: Si no se puede leer el archivo CSV
    """
    try:
        import pandas as pd
        
        # Intentar detectar el separador automáticamente
        separadores = [',', ';', '\t', '|']
        df = None
        separador_usado = None
        
        for sep in separadores:
            try:
                # Intentar leer con diferentes codificaciones
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df_temp = pd.read_csv(file_path, sep=sep, encoding=encoding)
                        # Si tiene más de una columna, probablemente sea el separador correcto
                        if len(df_temp.columns) > 1:
                            df = df_temp
                            separador_usado = sep
                            logger.info(f"CSV {file_path} cargado con separador '{sep}' y codificación '{encoding}'")
                            break
                    except (UnicodeDecodeError, pd.errors.EmptyDataError):
                        continue
                
                if df is not None:
                    break
                    
            except Exception:
                continue
        
        # Si no se pudo detectar automáticamente, usar coma por defecto
        if df is None:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                separador_usado = ','
                logger.info(f"CSV {file_path} cargado con separador por defecto ','")
            except Exception:
                df = pd.read_csv(file_path, encoding='latin-1')
                separador_usado = ','
                logger.info(f"CSV {file_path} cargado con separador ',' y codificación 'latin-1'")
        
        if df.empty:
            raise ValueError("El archivo CSV está vacío")
        
        # Convertir a texto estructurado
        text_parts = []
        
        # Añadir información del archivo
        text_parts.append(f"=== ARCHIVO CSV: {Path(file_path).name} ===")
        text_parts.append(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
        text_parts.append(f"Separador detectado: '{separador_usado}'")
        text_parts.append("")
        
        # Añadir nombres de columnas
        text_parts.append("=== COLUMNAS ===")
        for i, col in enumerate(df.columns, 1):
            text_parts.append(f"{i}. {col}")
        text_parts.append("")
        
        # Añadir estadísticas básicas si hay columnas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text_parts.append("=== ESTADÍSTICAS NUMÉRICAS ===")
            for col in numeric_cols:
                stats = df[col].describe()
                text_parts.append(f"Columna: {col}")
                text_parts.append(f"  - Promedio: {stats['mean']:.2f}")
                text_parts.append(f"  - Mediana: {stats['50%']:.2f}")
                text_parts.append(f"  - Mínimo: {stats['min']:.2f}")
                text_parts.append(f"  - Máximo: {stats['max']:.2f}")
            text_parts.append("")
        
        # Añadir una muestra de los datos (primeras 10 filas)
        text_parts.append("=== MUESTRA DE DATOS (Primeras 10 filas) ===")
        sample_size = min(10, len(df))
        for i, row in df.head(sample_size).iterrows():
            text_parts.append(f"Fila {i + 1}:")
            for col in df.columns:
                value = str(row[col])
                # Truncar valores muy largos
                if len(value) > 100:
                    value = value[:97] + "..."
                text_parts.append(f"  - {col}: {value}")
            text_parts.append("")
        
        # Si hay más de 10 filas, mencionar que hay más datos
        if len(df) > 10:
            text_parts.append(f"... y {len(df) - 10} filas adicionales")
            text_parts.append("")
        
        # Añadir resumen de tipos de datos
        text_parts.append("=== TIPOS DE DATOS ===")
        for col, dtype in df.dtypes.items():
            text_parts.append(f"{col}: {str(dtype)}")
        
        full_text = "\n".join(text_parts)
        logger.info(f"CSV {file_path} procesado: {len(df)} filas, {len(df.columns)} columnas")
        return full_text
        
    except ImportError:
        raise ImportError("pandas no está instalado. Instala con: pip install pandas")
    except Exception as e:
        logger.error(f"Error cargando archivo CSV {file_path}: {e}")
        raise


def load_document(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Carga un documento de cualquier tipo soportado.
    
    Args:
        file_path (str): Ruta del archivo a cargar
        
    Returns:
        Tuple[str, Dict[str, Any]]: (contenido, metadata)
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el tipo de archivo no es soportado
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"El archivo {file_path} no existe")
    
    # Metadata básica
    file_stat = file_path.stat()
    metadata = {
        "file_size": file_stat.st_size,
        "file_extension": file_path.suffix.lower(),
        "file_path": str(file_path.absolute()),
        "file_name": file_path.name,
        "created_time": file_stat.st_ctime,
        "modified_time": file_stat.st_mtime
    }
    
    # Cargar según el tipo de archivo
    extension = file_path.suffix.lower()
    
    if extension == '.txt':
        content = load_text_file(str(file_path))
        metadata["file_type"] = "text"
    elif extension == '.pdf':
        content = load_pdf_file(str(file_path))
        metadata["file_type"] = "pdf"
    elif extension in ['.docx', '.doc']:
        if extension == '.doc':
            logger.warning(f"Archivo .DOC detectado: {file_path}. Considera convertirlo a .DOCX para mejor compatibilidad")
        content = load_docx_file(str(file_path))
        metadata["file_type"] = "word"
    elif extension == '.csv':
        content = load_csv_file(str(file_path))
        metadata["file_type"] = "csv"
    else:
        # Intentar cargar como texto plano
        try:
            content = load_text_file(str(file_path))
            metadata["file_type"] = "text"
            logger.warning(f"Archivo {file_path} tratado como texto plano")
        except Exception:
            raise ValueError(f"Tipo de archivo no soportado: {extension}")
    
    # Añadir estadísticas del contenido
    metadata["content_length"] = len(content)
    metadata["word_count"] = len(content.split())
    metadata["line_count"] = len(content.splitlines())
    
    # Añadir información sobre el contenido
    metadata["is_empty"] = len(content.strip()) == 0
    metadata["has_special_chars"] = any(ord(char) > 127 for char in content[:1000])  # Verificar primeros 1000 chars
    
    logger.info(f"Documento cargado: {metadata['word_count']} palabras, {metadata['content_length']} caracteres")
    
    return content, metadata


def get_supported_file_types() -> List[str]:
    """
    Retorna la lista de tipos de archivo soportados.
    
    Returns:
        List[str]: Lista de extensiones soportadas
    """
    return ['.txt', '.pdf', '.docx', '.doc', '.csv']


def get_file_type_description(extension: str) -> str:
    """
    Obtiene la descripción de un tipo de archivo.
    
    Args:
        extension (str): Extensión del archivo
        
    Returns:
        str: Descripción del tipo de archivo
    """
    descriptions = {
        '.txt': 'Archivos de texto plano',
        '.pdf': 'Documentos PDF',
        '.docx': 'Documentos Word (DOCX)',
        '.doc': 'Documentos Word (DOC)',
        '.csv': 'Archivos CSV (Comma Separated Values)'
    }
    return descriptions.get(extension.lower(), 'Archivo de texto')


def validate_file_for_processing(file_path: str) -> Dict[str, Any]:
    """
    Valida si un archivo puede ser procesado.
    
    Args:
        file_path (str): Ruta del archivo
        
    Returns:
        Dict[str, Any]: Resultado de la validación
    """
    result = {
        "valid": False,
        "error": None,
        "warning": None,
        "file_info": {}
    }
    
    try:
        file_path = Path(file_path)
        
        # Verificar existencia
        if not file_path.exists():
            result["error"] = "El archivo no existe"
            return result
        
        # Verificar que es un archivo
        if not file_path.is_file():
            result["error"] = "La ruta no es un archivo"
            return result
        
        # Verificar extensión soportada
        extension = file_path.suffix.lower()
        if extension not in get_supported_file_types():
            result["error"] = f"Tipo de archivo no soportado: {extension}"
            return result
        
        # Verificar tamaño
        file_size = file_path.stat().st_size
        max_size = 100 * 1024 * 1024  # 100 MB
        
        if file_size == 0:
            result["warning"] = "El archivo está vacío"
        elif file_size > max_size:
            result["warning"] = f"El archivo es muy grande ({file_size / 1024 / 1024:.1f} MB). Puede tardar en procesarse"
        
        result["file_info"] = {
            "size": file_size,
            "extension": extension,
            "name": file_path.name
        }
        
        result["valid"] = True
        
    except Exception as e:
        result["error"] = f"Error validando archivo: {str(e)}"
    
    return result


def clean_text(text: str) -> str:
    """
    Limpia y normaliza el texto.
    
    Args:
        text (str): Texto a limpiar
        
    Returns:
        str: Texto limpio
    """
    if not text:
        return ""
    
    # Reemplazar múltiples espacios en blanco con uno solo
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remover espacios al inicio y final
    text = text.strip()
    
    # Normalizar saltos de línea
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remover múltiples saltos de línea consecutivos
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def extract_text_statistics(text: str) -> Dict[str, Any]:
    """
    Extrae estadísticas del texto.
    
    Args:
        text (str): Texto a analizar
        
    Returns:
        Dict[str, Any]: Estadísticas del texto
    """
    if not text:
        return {
            "character_count": 0,
            "word_count": 0,
            "line_count": 0,
            "paragraph_count": 0,
            "average_words_per_line": 0,
            "has_content": False
        }
    
    lines = text.splitlines()
    words = text.split()
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    return {
        "character_count": len(text),
        "word_count": len(words),
        "line_count": len(lines),
        "paragraph_count": len(paragraphs),
        "average_words_per_line": len(words) / len(lines) if lines else 0,
        "has_content": len(text.strip()) > 0
    }