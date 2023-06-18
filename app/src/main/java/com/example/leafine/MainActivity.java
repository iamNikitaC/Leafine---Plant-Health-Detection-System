package com.example.leafine;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.content.DialogInterface;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.leafine.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;


public class MainActivity extends AppCompatActivity {


    private ImageView imgView;
    private TextView tv;
    private Bitmap img;


    private String getClassName(float[] output) {
        String[] classNames = {
                "Diseased Apple Plant, Name of disease: Apple_scab",
                "Diseased Apple Plant, Name of disease: Black_rot",
                "Healthy Apple Plant",
                "Diseased Corn_(maize) Plant, Name of disease: Common_rust",
                "Diseased Corn_(maize) Plant, Name of disease: Northern_Leaf_Blight",
                "Healthy Corn_(maize) Plant",
                "Diseased Grape Plant, Name of disease: Black_rot",
                "Diseased Grape Plant, Name of disease: Leaf_Blight_Isariopsis_Leaf_spot",
                "Healthy Grape Plant"};


        int maxIndex = 0;
        float maxValue = output[0];

        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }

        return classNames[maxIndex];
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgView = (ImageView) findViewById(R.id.imageView);
        tv = (TextView) findViewById(R.id.textView);
        Button select = (Button) findViewById(R.id.button);
        Button predict = (Button) findViewById(R.id.button2);


        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);

            }
        });


        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                img = Bitmap.createScaledBitmap(img, 224, 224, true);

                try {
                    Model model = Model.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

                    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                    tensorImage.load(img);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                    
                    model.close();

                    tv.setText("The predicted class is: " + getClassName(outputFeature0.getFloatArray()));
                    //tv.setText(outputFeature0.getFloatArray()[0] + "\n"+outputFeature0.getFloatArray()[1] + "\n"+outputFeature0.getFloatArray()[2] + "\n"+outputFeature0.getFloatArray()[3] + "\n"+outputFeature0.getFloatArray()[4] + "\n"+outputFeature0.getFloatArray()[5] + "\n"+outputFeature0.getFloatArray()[6] + "\n"+outputFeature0.getFloatArray()[7] + "\n"+outputFeature0.getFloatArray()[8]);

                } catch (IOException e) {

                }

            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 100)
        {
            imgView.setImageURI(data.getData());

            Uri uri = data.getData();
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}