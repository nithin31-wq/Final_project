import torch
import torch.nn as nn

class CNNAutoencoderLSTM(nn.Module):
    """
    Enhanced CNN-LSTM Autoencoder:
    - Conv1d encoder (two blocks)
    - LSTM encoder
    - Fully connected latent
    - LSTM decoder + Conv1d decoder
    """
    def __init__(self, input_dim, cnn_filters=64, lstm_hidden=128, latent_dim=128, bidirectional=False, dropout=0.2):
        super(CNNAutoencoderLSTM, self).__init__()
        self.input_dim = input_dim
        self.cnn_filters = cnn_filters
        self.bidirectional = bidirectional
        self.lstm_hidden = lstm_hidden
        self.latent_dim = latent_dim
        self.dropout = dropout

        # Encoder conv blocks
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # Second conv block
        self.encoder_cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters * 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=cnn_filters * 2, out_channels=cnn_filters * 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # LSTM encoder
        num_dirs = 2 if bidirectional else 1
        self.encoder_lstm = nn.LSTM(input_size=cnn_filters * 2, hidden_size=lstm_hidden, batch_first=True, bidirectional=bidirectional)
        self.fc_enc = nn.Linear(lstm_hidden * num_dirs, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, lstm_hidden * num_dirs)
        self.decoder_lstm = nn.LSTM(input_size=lstm_hidden * num_dirs, hidden_size=cnn_filters * 2, batch_first=True)
        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=cnn_filters * 2, out_channels=cnn_filters, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=cnn_filters, out_channels=input_dim, kernel_size=3, padding=1)
        )

        # Optional normalization
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        x: [batch, seq_len, features]
        returns: recon [batch, seq_len, features], latent [batch, latent_dim]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]

        # Permute for Conv1d: [batch, features, seq_len]
        x_perm = x.permute(0, 2, 1)

        # Encoder conv
        h = self.encoder_cnn(x_perm)
        h = self.encoder_cnn2(h)

        # Permute for LSTM: [batch, seq_len, channels]
        h_perm = h.permute(0, 2, 1)

        _, (h_n, _) = self.encoder_lstm(h_perm)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]

        latent = self.fc_enc(h_last)  # [batch, latent_dim]

        # Decoder: expand latent to seq
        dec_in = self.fc_dec(latent).unsqueeze(1).repeat(1, x.shape[1], 1)  # [batch, seq_len, hidden']
        dec_out, _ = self.decoder_lstm(dec_in)
        dec_out = dec_out.permute(0, 2, 1)  # [batch, channels, seq_len]
        recon = self.decoder_cnn(dec_out)   # [batch, features, seq_len]
        recon = recon.permute(0, 2, 1)      # [batch, seq_len, features]

        recon = self.layer_norm(recon)
        return recon, latent
