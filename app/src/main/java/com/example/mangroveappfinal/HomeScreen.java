package com.example.mangroveappfinal;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

public class HomeScreen extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home_screen);
    }

    public void launchStart(View v) {
        //launch a new activity
        Intent i = new Intent(this, MainActivity.class);
        startActivity(i);

    }

    public void launchInfoCare(View v) {
        //launch a new activity
        Intent i = new Intent(this, InfoManagement.class);
        startActivity(i);

    }


    public void launchAbout1(View v) {
        //launch a new activity
        Intent i = new Intent(this, About.class);
        startActivity(i);

    }

    public void launchData(View v) {
        //launch a new activity
        Intent i = new Intent(this, ShowActivity.class);
        startActivity(i);

    }
    public void showSummary(View v) {
        //launch a new activity
        Intent i = new Intent(this, MonthlySummary.class);
        startActivity(i);

    }



    int counter = 0;

    @Override
    public void onBackPressed() {
        counter++;
        if (counter == 1)
            Toast.makeText(getApplicationContext(), "Click Again To Exit.", Toast.LENGTH_SHORT).show();
        if (counter == 2)
            super.onBackPressed();

    }
}

