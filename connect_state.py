# Abstract
from environment_state import EnvironmentState

# Types
from typing import Optional, List, Any

# Libraries
import numpy as np
import matplotlib.pyplot as plt


class ConnectState(EnvironmentState):
    """
    Environment state representation for the Connect Four game.
    """

    def __init__(self, board: Optional[np.ndarray] = None): # Initializes the Connect Four game state. Has a NumPy array representing the board state
        self.ROWS = 6 # 6 filas
        self.COLS = 7 # 7 columnas

        if board is None: 
            self.board = np.zeros((self.ROWS, self.COLS), dtype=int) # If None, an empty board is created
        else:
            self.board = board.copy()

    def is_final(self) -> bool:
        is_winner = self.get_winner() != 0 # Verifica si hay un ganador. 0 = No hay ganador
        full_board = np.all(self.board != 0) # Verifica si el tablero está vacío. True si todas las celdas son distintas de 0

        return is_winner or full_board
    
    def is_applicable(self, event: Any) -> bool:
        col = int(event)

        if not (0 <= col < self.COLS): # Verifica si la columna está dentro del rango permitido
            return False
        space = self.board[0, col] == 0  # Si space = 0, hay espacio para colocar una ficha

        return space

    def transition(self, col: int) -> "EnvironmentState":
        if not self.is_applicable(col): # Verifica si la jugada es valida
            raise ValueError("Invalid move")
        
        row = self.ROWS - 1
        while self.board[row, col] != 0: # Si la posición está ocupada sigue buscando una vacía (0)
            row -= 1

        tiles = np.count_nonzero(self.board) # Cantidad de fichas
        player = -1 if tiles % 2 == 0 else 1 # Determina el jugador actual

        current_board = self.board.copy()
        current_board[row, col] = player # Actualiza el tablero

        return ConnectState(current_board) # Cambio en el estado

    def get_winner(self) -> int: # Determines the winner in the current state
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Direcciones para ganar

        for row in range(self.ROWS): # Recorrer filas
            for col in range(self.COLS): # Recorrer columnas
                current_cell = self.board[row, col]
                if current_cell == 0:
                    continue
                for delta_row, delta_col in directions:
                    consecutive_count = 0
                    for step in range(4): # Revisa 4 celdas seguidas en esa dirección
                        next_row = row + delta_row * step
                        next_col = col + delta_col * step

                        if 0 <= next_row < self.ROWS and 0 <= next_col < self.COLS: # Verifica que la posición esté dentro del tablero
                            if self.board[next_row, next_col] == current_cell:
                                consecutive_count += 1
                            else:
                                break  
                        else:
                            break
                    if consecutive_count == 4: # Si se encuentran 4 seguidas, hay ganador
                        return current_cell

        return 0 # Returns -1 if red has won, 1 if yellow has won, 0 if no winner

    def is_col_free(self, col: int) -> bool:
        space = self.board[0, col] == 0  # Si space = 0, hay espacio para colocar una ficha

        return space # Returns True if the column has space for a tile

    def get_heights(self) -> List[int]:
        heights = []
        for col in range(self.COLS): # Recorre cada columna del tablero
            full = np.count_nonzero(self.board[:, col]) # Cuenta las celdas en esa columna que no son cero, tienen ficha
            heights.append(full)

        return heights # Returns a list of integers indicating the number of tiles per column

    def get_free_cols(self) -> List[int]:
        free_columns = [col for col in range(self.COLS) if self.is_col_free(col)]

        return free_columns # Returns indices of columns with at least one free cell

    def show(self, size: int = 1500, ax: Optional[plt.Axes] = None) -> None:
        """
        Visualizes the current board state using matplotlib.

        Parameters
        ----------
        size : int, optional
            Size of the stones, by default 1500.
        ax : Optional[matplotlib.axes._axes.Axes], optional
            Axes to plot on. If None, a new figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        pos_red = np.where(self.board == -1)
        pos_yellow = np.where(self.board == 1)

        ax.scatter(pos_yellow[1] + 0.5, 5.5 - pos_yellow[0], color="yellow", s=size)
        ax.scatter(pos_red[1] + 0.5, 5.5 - pos_red[0], color="red", s=size)

        ax.set_ylim([0, self.board.shape[0]])
        ax.set_xlim([0, self.board.shape[1]])
        ax.grid()

        if fig is not None:
            plt.show()

# TEST

if __name__ == "__main__":

    state = ConnectState() # Crear el estado inicial

    print("Board:")
    print(state.board)
    print("")

    moves = [2, 2, 3, 3, 4, 4, 1]  # Jugadas alternando a los jugadores
    for move in moves:
        if state.is_applicable(move):
            state = state.transition(move)
            print(f"Tile placed in column {move}")
        else:
            print(f"Tile can't be placed in column {move}")
    print("")

    print("Board:") # Tablero actualizado
    print(state.board)
    print("")

    winner = state.get_winner() # Verificar si hay un ganador
    if winner  == -1:
        print("Red is the winner (-1)")
    elif winner == 1:
        print("Yellow is the winner (1)")
    else:
        print("There is no winner yet")

    state.show() # Mostrar la interfaz